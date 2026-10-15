"""LangGraph node implementations: planner, searcher, matcher, mailer."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator
from tavily import TavilyClient

from . import email_service
from .config import Settings, get_settings
from .state import JobHunterState

logger = logging.getLogger(__name__)

MODEL_TEMPERATURE = 0.2
_SNIPPET_LIMIT = 900
_RESUME_LIMIT = 6000
_JOB_BOARD_SITES = (
    "site:greenhouse.io",
    "site:lever.co",
    "site:myworkdayjobs.com",
    "site:linkedin.com/jobs",
    "site:jobstreet.",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def _date_label() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y")


def _safe_node(name: str) -> Callable:
    """Wrap a node so any exception is logged and recorded instead of raised."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> JobHunterState:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - node boundary must never crash the run
                logger.exception("Node '%s' failed", name)
                state: JobHunterState = dict(args[0]) if args else {}
                state["error"] = f"{name}: {type(exc).__name__}: {exc}"
                return state

        return wrapper

    return decorator


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response (fences/notes tolerated)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("[") if "[" in cleaned else cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in model output: {text[:200]!r}")

    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(cleaned[start:])
    return value


def _llm(settings: Settings, model_id: str | None = None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_id or settings.gemini_model,
        temperature=MODEL_TEMPERATURE,
        api_key=settings.gemini_api_key,
        timeout=60,
        max_retries=2,
    )


# Google retires Gemini models periodically (e.g. gemini-1.5-flash 404s, gemini-2.0-flash
# shut down June 2026), so try the configured model first, then current flash models.
_MODEL_FALLBACKS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)
_active_model: str | None = None


def _is_model_not_found(exc: Exception) -> bool:
    text = str(exc)
    return "404" in text or "NOT_FOUND" in text or "is not found" in text.lower()


def _candidate_models(settings: Settings) -> list[str]:
    candidates: list[str] = []
    for model_id in (_active_model, settings.gemini_model, *_MODEL_FALLBACKS):
        if model_id and model_id not in candidates:
            candidates.append(model_id)
    return candidates


def _generate(settings: Settings, messages: list) -> Any:
    """Invoke Gemini, falling through the model chain on NOT_FOUND errors."""
    global _active_model

    last_error: Exception | None = None
    for model_id in _candidate_models(settings):
        try:
            response = _llm(settings, model_id).invoke(messages)
        except Exception as exc:  # noqa: BLE001 - decide fallback by error text
            if not _is_model_not_found(exc):
                raise
            logger.warning("Gemini model %s unavailable, trying next: %s", model_id, exc)
            last_error = exc
            continue
        if model_id != _active_model:
            logger.info("Using Gemini model %s", model_id)
        _active_model = model_id
        return response

    raise last_error or RuntimeError("no usable Gemini model found")


def _fallback_queries(target_roles: str, target_locations: str) -> list[str]:
    """Deterministic queries used when the planner model is unavailable."""
    roles = [r.strip() for r in target_roles.split(",") if r.strip()] or [
        "Software Engineer"
    ]
    location = target_locations.strip() or "Remote"
    return [
        f'"{roles[i % len(roles)]}" jobs "{location}" {_JOB_BOARD_SITES[i % len(_JOB_BOARD_SITES)]}'
        for i in range(3)
    ]


def _source_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except ValueError:
        return "unknown"


# --------------------------------------------------------------------------- #
# Structured scoring schema
# --------------------------------------------------------------------------- #
class JobMatchScore(BaseModel):
    match_score: int = Field(ge=0, le=100, description="Fit score from 0 to 100")
    rationale: str = Field(description="Two-sentence explanation of the fit")
    matched_skills: list[str] = Field(default_factory=list)
    missing_or_gap_skills: list[str] = Field(default_factory=list)

    @field_validator("match_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: Any) -> int:
        return max(0, min(100, int(round(float(value)))))

    @field_validator("rationale", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        return str(value or "").strip()


# --------------------------------------------------------------------------- #
# Node 1: query planner
# --------------------------------------------------------------------------- #
@_safe_node("planner")
def planner(
    state: JobHunterState,
    settings: Settings | None = None,
) -> JobHunterState:
    """Generate 3 targeted job-board search strings with Gemini."""
    cfg = _resolve(settings)
    roles = state.get("target_roles") or cfg.target_roles
    locations = state.get("target_locations") or cfg.target_locations
    resume = (state.get("resume_text") or "").strip()

    prompt = [
        SystemMessage(
            content=(
                "You are a job-search query planner for technical candidates. "
                "You produce search strings that surface ACTIVE job postings, not "
                "career advice, blog posts, or stale listings."
            )
        ),
        HumanMessage(
            content=(
                "Target roles: {roles}\n"
                "Target locations: {locations}\n\n"
                "Candidate resume:\n{resume}\n\n"
                "Return ONLY a JSON array of exactly 3 search strings, each tailored "
                "to a different slice of the candidate's background. Rules:\n"
                "1. Each query must combine a specific technical job title with a location.\n"
                "2. Spread the queries across job boards using operators such as "
                "{boards} plus regional boards (JobStreet, LinkedIn Jobs) where relevant.\n"
                "3. Prefer quoted phrases and recent-active-posting phrasing, e.g. "
                '"Senior Python Engineer" "Singapore" site:greenhouse.io hiring 2026.\n'
                "4. No commentary, no markdown fences - raw JSON array only."
            ).format(
                roles=roles,
                locations=locations,
                resume=resume[:_RESUME_LIMIT] or "(not provided)",
                boards=", ".join(_JOB_BOARD_SITES),
            )
        ),
    ]

    try:
        response = _message_text(_generate(cfg, prompt))
        queries = _extract_json(response)
        if not isinstance(queries, list):
            raise ValueError("model did not return a list")
        queries = [str(q).strip() for q in queries if str(q).strip()][:3]
    except Exception as exc:
        logger.warning("Planner fell back to deterministic queries: %s", exc)
        queries = []

    if len(queries) < 3:
        for fallback in _fallback_queries(roles, locations):
            if fallback not in queries:
                queries.append(fallback)
            if len(queries) == 3:
                break

    logger.info("Planner produced %d queries", len(queries))
    state["queries"] = queries[:3]
    return state


# --------------------------------------------------------------------------- #
# Node 2: Tavily web search
# --------------------------------------------------------------------------- #
@_safe_node("searcher")
def searcher(
    state: JobHunterState,
    settings: Settings | None = None,
) -> JobHunterState:
    """Run each query through Tavily and dedupe results by URL."""
    if state.get("error") or not state.get("queries"):
        state.setdefault("raw_jobs", [])
        return state

    cfg = _resolve(settings)
    client = TavilyClient(api_key=cfg.tavily_api_key)
    max_results = cfg.max_results_per_query

    seen: dict[str, dict] = {}
    failures = 0
    for query in state["queries"]:
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
            )
        except Exception as exc:
            failures += 1
            logger.warning("Tavily search failed for %r: %s", query, exc)
            continue

        for result in response.get("results", []) or []:
            url = (result.get("url") or "").strip()
            if not url or url in seen:
                continue
            snippet = (
                result.get("content") or result.get("raw_content") or ""
            ).strip()
            seen[url] = {
                "title": (result.get("title") or "Untitled role").strip(),
                "url": url,
                "snippet": snippet[:_SNIPPET_LIMIT],
                "source": _source_from_url(url),
            }

    state["raw_jobs"] = list(seen.values())
    if failures and failures == len(state["queries"]):
        state["error"] = f"searcher: all {failures} Tavily queries failed"
        logger.error("%s", state["error"])
    logger.info("Searcher collected %d unique postings", len(state["raw_jobs"]))
    return state


# --------------------------------------------------------------------------- #
# Node 3: matcher / scorer
# --------------------------------------------------------------------------- #
def _scoring_messages(resume: str, jobs: list[dict]) -> list:
    catalogue = [
        {
            "id": index,
            "title": job.get("title", ""),
            "source": job.get("source", ""),
            "snippet": (job.get("snippet") or "")[:_SNIPPET_LIMIT],
        }
        for index, job in enumerate(jobs)
    ]
    return [
        SystemMessage(
            content=(
                "You are a strict technical recruiter scoring job postings against a "
                "candidate's resume. Score only on evidence present in the resume: "
                "skills, seniority, domain, and location. Be conservative - a weak "
                "posting should score below 50, never above 60."
            )
        ),
        HumanMessage(
            content=(
                "CANDIDATE RESUME:\n{resume}\n\n"
                "JOB POSTINGS (JSON):\n{jobs}\n\n"
                "For EVERY job id, return ONLY a JSON array (no markdown fences) of objects with:\n"
                '  "id": the original integer id,\n'
                '  "match_score": integer 0-100,\n'
                '  "rationale": exactly 2 sentences explaining the fit,\n'
                '  "matched_skills": array of skills from the resume that the posting requires,\n'
                '  "missing_or_gap_skills": array of required skills the resume lacks.\n'
                "Include every id exactly once."
            ).format(
                resume=resume[:_RESUME_LIMIT] or "(no resume provided)",
                jobs=json.dumps(catalogue, ensure_ascii=False),
            )
        ),
    ]


def _parse_scores(raw: str, expected_ids: set[int]) -> dict[int, JobMatchScore]:
    data = _extract_json(raw)
    if isinstance(data, dict):
        data = data.get("jobs") or data.get("results") or list(data.values())
    if not isinstance(data, list):
        raise ValueError("scoring response was not a list")

    parsed: dict[int, JobMatchScore] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            job_id = int(entry.get("id", entry.get("job_id", -1)))
        except (TypeError, ValueError):
            continue
        if job_id not in expected_ids or job_id in parsed:
            continue
        entry = {k: v for k, v in entry.items() if k in JobMatchScore.model_fields}
        parsed[job_id] = JobMatchScore.model_validate(entry)
    return parsed


@_safe_node("matcher")
def matcher(
    state: JobHunterState,
    settings: Settings | None = None,
) -> JobHunterState:
    """Score every posting against the resume and keep only strong matches."""
    if state.get("error") or not state.get("raw_jobs"):
        state.setdefault("matches", [])
        return state

    cfg = _resolve(settings)
    jobs: list[dict] = state["raw_jobs"]
    expected_ids = set(range(len(jobs)))

    scores: dict[int, JobMatchScore] = {}
    try:
        response = _message_text(
            _generate(cfg, _scoring_messages(state.get("resume_text", ""), jobs))
        )
        scores = _parse_scores(response, expected_ids)
        if len(scores) < len(jobs):
            raise ValueError(f"batch scored {len(scores)}/{len(jobs)} jobs")
    except Exception as exc:
        logger.warning("Batch scoring unavailable (%s); retrying per job", exc)
        scores = {}

    for index, job in enumerate(jobs):
        if index in scores:
            continue
        try:
            response = _message_text(
                _generate(cfg, _scoring_messages(state.get("resume_text", ""), [job]))
            )
            individual = _parse_scores(response, {index})
            if index in individual:
                scores[index] = individual[index]
            else:
                logger.warning("No score returned for %r", job.get("title"))
        except Exception as exc:
            logger.warning("Scoring failed for %r: %s", job.get("title"), exc)

    matches: list[dict] = []
    for index, job in enumerate(jobs):
        score = scores.get(index)
        if score is None:
            continue
        if score.match_score < cfg.match_threshold:
            continue
        matches.append(
            {
                "title": job.get("title", "Untitled role"),
                "url": job.get("url", ""),
                "match_score": score.match_score,
                "rationale": score.rationale,
                "key_skills_matched": score.matched_skills,
                "missing_skills": score.missing_or_gap_skills,
            }
        )

    matches.sort(key=lambda item: item["match_score"], reverse=True)
    state["matches"] = matches
    if jobs and not scores:
        state["error"] = "matcher: no postings could be scored"
        logger.error("%s", state["error"])
    logger.info(
        "Matcher kept %d/%d postings at or above %d%%",
        len(matches),
        len(jobs),
        cfg.match_threshold,
    )
    return state


# --------------------------------------------------------------------------- #
# Node 4: HTML digest dispatch
# --------------------------------------------------------------------------- #
@_safe_node("mailer")
def mailer(
    state: JobHunterState,
    settings: Settings | None = None,
) -> JobHunterState:
    """Render the digest and send it over SMTP_SSL."""
    cfg = _resolve(settings)
    matches = state.get("matches") or []
    date_label = _date_label()
    roles = state.get("target_roles") or cfg.target_roles

    if matches:
        html_body = email_service.render_matches_email(
            matches, date_label=date_label, roles=roles
        )
        subject = f"[Job Agent] {date_label}: {len(matches)} match{'es' if len(matches) != 1 else ''}"
    else:
        html_body = email_service.render_empty_email(date_label, roles)
        subject = f"[Job Agent] {date_label}: no high-confidence roles"

    try:
        state["email_status"] = email_service.send_email(
            cfg, subject, html_body, date_label=date_label
        )
    except Exception as exc:
        logger.exception("Email dispatch failed")
        state["email_status"] = f"failed: {type(exc).__name__}: {exc}"
        state["error"] = f"mailer: {type(exc).__name__}: {exc}"

    return state
