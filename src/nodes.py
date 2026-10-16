"""LangGraph node implementations: planner, searcher, matcher, mailer."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
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
_QUERY_COUNT = 5
# Ordered by how well each board carries postings for a specific country. JobStreet
# dominates Malaysia/Singapore and the Gulf; Greenhouse/Lever/Workday are US/EU-heavy
# and mostly waste queries when the target locations are a single non-US country.
# The planner is told to treat this as priority order, so the country-local boards
# come first.
_JOB_BOARD_SITES = (
    "site:my.jobstreet.com",
    "site:jobstreet.com.my",
    "site:linkedin.com/jobs",
    "site:myworkdayjobs.com",
    "site:lever.co",
    "site:greenhouse.io",
)

# Geographic expansions so "Kuala Lumpur" also accepts jobs posted in the wider
# Klang Valley tech corridor, and "Penang" accepts the island's tech belt.
_REGION_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "kuala lumpur": (
        "kuala lumpur",
        "klcc",
        "bangsar",
        "petaling jaya",
        "selangor",
        "cyberjaya",
        "shah alam",
        "subang jaya",
        "klang valley",
        "greater kl",
        "mont kiara",
        "putrajaya",
        "ampang",
        "cheras",
        "kota damansara",
    ),
    "penang": (
        "penang",
        "george town",
        "georgetown",
        "bayan lepas",
        "kulim",
        "seberang perai",
        "butterworth",
        "tanjung bungah",
    ),
}


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
        max_retries=0,
    )


# Google retires Gemini models periodically (e.g. gemini-1.5-flash 404s, gemini-2.0-flash
# shut down June 2026), so try the configured model first, then current flash models.
_MODEL_FALLBACKS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)
_active_model: str | None = None
# The free tier also caps generate_content at 20 requests/DAY per model
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier). That cap is per model, so a
# model that reports a hours-long retryDelay is dead for the day and the chain must
# move on instead of sleeping; models known-dead this run are skipped outright.
_quota_exhausted_models: set[str] = set()
# Hard per-run budget so the matcher's per-job fallback cannot burn the daily quota
# on a single run (one request per job adds up fast).
_DAILY_REQUEST_BUDGET = max(1, int(os.getenv("GEMINI_REQUEST_BUDGET", "16")))
_requests_used = 0


def _acquire_request_slot() -> None:
    global _requests_used
    if _requests_used >= _DAILY_REQUEST_BUDGET:
        raise RuntimeError(
            f"Gemini request budget exhausted for this run ({_requests_used}/{_DAILY_REQUEST_BUDGET})"
        )
    _requests_used += 1


def _is_model_not_found(exc: Exception) -> bool:
    text = str(exc)
    return "404" in text or "NOT_FOUND" in text or "is not found" in text.lower()


_RETRY_AFTER_RE = re.compile(r"retry(?:Delay)?[:\s]+(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE)
_MAX_RATE_LIMIT_SLEEP = 45.0


def _rate_limit_delay(exc: Exception) -> float | None:
    """Seconds to wait for a 429/503, or None if the error is not rate limiting.

    Gemini's free tier caps generate_content at ~5 requests/minute, which the
    batch-then-per-job matcher can exceed. Honoring the server's retryDelay lets
    the run finish instead of scoring nothing.
    """
    text = str(exc)
    if not ("429" in text or "RESOURCE_EXHAUSTED" in text or "503" in text or "UNAVAILABLE" in text):
        return None
    match = _RETRY_AFTER_RE.search(text)
    if match:
        return min(float(match.group(1)) + 1.0, _MAX_RATE_LIMIT_SLEEP)
    return 10.0


def _is_retryable(exc: Exception) -> bool:
    text = str(exc)
    return _rate_limit_delay(exc) is not None or "timeout" in text.lower()


_RETRY_HOURS_RE = re.compile(r"retry in (\d+)h", re.IGNORECASE)
# A retryDelay beyond this is not "back off a few seconds" -- it is the daily
# quota saying come back tomorrow (observed: retryDelay: 47777s on free tier).
_DAILY_QUOTA_MIN_DELAY = 300.0


def _is_daily_quota(exc: Exception) -> bool:
    """True when a 429 means the per-model daily quota, not a burst limit."""
    text = str(exc)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return False
    hours = _RETRY_HOURS_RE.search(text)
    if hours and int(hours.group(1)) >= 1:
        return True
    match = _RETRY_AFTER_RE.search(text)
    return bool(match and float(match.group(1)) > _DAILY_QUOTA_MIN_DELAY)


def _is_quota_exhausted(exc: Exception) -> bool:
    """Daily quota gone for every model, or our own per-run budget spent."""
    text = str(exc).lower()
    return _is_daily_quota(exc) or "budget exhausted" in text or "no usable gemini" in text


def _candidate_models(settings: Settings) -> list[str]:
    candidates: list[str] = []
    for model_id in (_active_model, settings.gemini_model, *_MODEL_FALLBACKS):
        if model_id and model_id not in candidates and model_id not in _quota_exhausted_models:
            candidates.append(model_id)
    return candidates


def _generate(settings: Settings, messages: list) -> Any:
    """Invoke Gemini, falling through the model chain on NOT_FOUND or daily quota."""
    global _active_model

    _acquire_request_slot()
    if not _candidate_models(settings):
        raise RuntimeError("all Gemini models have exhausted their daily quota")

    last_error: Exception | None = None
    for model_id in _candidate_models(settings):
        for attempt in range(3):
            try:
                response = _llm(settings, model_id).invoke(messages)
            except Exception as exc:  # noqa: BLE001 - decide fallback by error text
                if _is_model_not_found(exc):
                    logger.warning(
                        "Gemini model %s unavailable, trying next: %s", model_id, exc
                    )
                    last_error = exc
                    break
                if _is_daily_quota(exc):
                    # Sleep-and-retry cannot help within this run; the cap is per model,
                    # so move to the next model in the chain instead of waiting hours.
                    _quota_exhausted_models.add(model_id)
                    logger.warning(
                        "Gemini daily quota exhausted for %s, trying next model: %s",
                        model_id,
                        exc,
                    )
                    last_error = exc
                    break
                delay = _rate_limit_delay(exc)
                if delay is None or attempt == 2:
                    raise
                logger.warning(
                    "Gemini rate limited (%s); waiting %.0fs (attempt %d/3)",
                    model_id,
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)
                continue
            else:
                if model_id != _active_model:
                    logger.info("Using Gemini model %s", model_id)
                _active_model = model_id
                return response
        continue

    raise last_error or RuntimeError("no usable Gemini model found")


def _fallback_queries(target_roles: str, target_locations: str) -> list[str]:
    """Deterministic queries used when the planner model is unavailable."""
    roles = [r.strip() for r in target_roles.split(",") if r.strip()] or [
        "Software Engineer"
    ]
    # Split on ';' so a multi-target string ("KL, Malaysia; Penang, Malaysia") becomes
    # one query per location -- quoting the whole raw string matches nothing. Commas
    # stay inside a location name ("Kuala Lumpur, Malaysia").
    locations = [loc.strip() for loc in target_locations.split(";") if loc.strip()] or [
        "Remote"
    ]
    return [
        f'"{roles[i % len(roles)]}" "{locations[i % len(locations)]}" '
        f"{_JOB_BOARD_SITES[i % len(_JOB_BOARD_SITES)]}"
        for i in range(_QUERY_COUNT)
    ]


def _source_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except ValueError:
        return "unknown"


# Career-site index pages and placeholder titles pollute results (e.g. a page whose
# <title> is literally "page_title", or a "/en" careers landing page).
_JUNK_TITLE_RE = re.compile(
    r"^(page_?title|jobs?|careers?|home|search( for jobs?)?|job( |_)application|"
    r"open positions|opportunities|vacanc(y|ies))\b"
    r"|\b(search for jobs|careers -|career site|job board|jobs in [A-Z])s?\b"
    r"|^jobs at\b",
    re.IGNORECASE,
)

# JobStreet (and similar aggregators) expose category landing pages at URLs that look
# like postings -- "/node-jobs-in-cheras", "/backend-engineer-jobs-in-kuala-lumpur" --
# but only contain listings, not requirements. A real JobStreet posting lives under a
# single /job/<slug>-<id> segment, so any path that is NOT that shape is a category page
# on an aggregator host and cannot be scored.
_JOB_SLUG_ID_RE = re.compile(r"^/job/[a-z0-9-]+-\d{4,}/?$", re.IGNORECASE)
# Other boards (Lever, Greenhouse, Workday) use their own posting paths; these are the
# shapes known to be a single posting rather than a listing index.
_BOARD_POSTING_RE = re.compile(
    r"^/(?:jobs?|positions?|openings?)/(?:view/)?[a-z0-9-]*\d{4,}/?$",
    re.IGNORECASE,
)
_AGGREGATOR_HOST_RE = re.compile(
    r"(?:^|\.)jobstreet\.|linkedin\.com$|indeed\.|glassdoor\.",
    re.IGNORECASE,
)
# A URL that unambiguously points at one posting. On job-board hosts the path segment
# names are fixed ("jobs/<id>", "job/<slug>-<id>", "<company>/jobs/<id>"), so a match
# here is a posting regardless of what the page's boilerplate <title> says -- Greenhouse
# titles every posting "Job Application for <Role>", and that title must not veto the URL.
_EXPLICIT_POSTING_RE = re.compile(
    r"job-boards(?:\.[a-z]{2,6})?\.greenhouse\.io/[^/]+/jobs/\d+"
    r"|jobs\.greenhouse\.io/[^/]+/jobs/\d+"
    r"|jobs\.lever\.co/[^/]+/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|/job/[a-z0-9-]+-\d{4,}"
    r"|/jobs/view/\d+"
    r"|myworkdayjobs\.com/[^/]+/job/[^/]+",
    re.IGNORECASE,
)


def _looks_like_posting(url: str, title: str, snippet: str) -> bool:
    """Heuristic filter for real job postings vs. career-site index pages."""
    cleaned = re.sub(r"\s+", " ", title).strip()
    try:
        parsed = urlparse(url)
        segments = [seg for seg in parsed.path.split("/") if seg]
    except ValueError:
        return False
    # A URL that names one posting beats any boilerplate <title>. Greenhouse titles
    # every posting "Job Application for <Role>" and many boards title a posting
    # "Jobs at <Company>", which the junk-title check would otherwise reject.
    if _EXPLICIT_POSTING_RE.search(url):
        return len(snippet) >= 30
    if not cleaned or _JUNK_TITLE_RE.search(cleaned):
        return False
    # Real postings sit at least two path levels deep (/jobs/123, /job/title-slug);
    # index pages are the domain root or a single locale/marketing segment.
    if len(segments) < 2:
        return False
    # Aggregators (JobStreet, LinkedIn, Indeed) publish category pages at paths that
    # look like postings. On those hosts only a single known posting shape is trusted;
    # elsewhere the two-segment rule above already filtered the index pages.
    if _AGGREGATOR_HOST_RE.search(parsed.netloc) and not (
        _JOB_SLUG_ID_RE.match(parsed.path) or _BOARD_POSTING_RE.match(parsed.path)
    ):
        return False
    return len(snippet) >= 30


def _title_key(title: str) -> str:
    """Normalize a title so near-duplicates collapse (spacing, case, punctuation)."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _location_signals(target_locations: str) -> list[str]:
    """Expand a location string into matchable signals.

    ``"Kuala Lumpur, Malaysia; Penang, Malaysia"`` becomes the literal parts plus
    the Klang Valley / Penang tech-belt expansions from `_REGION_EXPANSIONS`.
    """
    parts = [
        part.strip().lower()
        for part in re.split(r"[,;]", target_locations or "")
        if part.strip()
    ]
    signals: list[str] = []

    def add(value: str) -> None:
        value = value.strip().lower()
        if value and value not in signals:
            signals.append(value)

    for part in parts:
        add(part)
        for region, expansions in _REGION_EXPANSIONS.items():
            if region in part:
                for expansion in expansions:
                    add(expansion)
    return signals


def _mentions_location(text: str, signals: list[str]) -> bool:
    """True when any signal appears in ``text`` (word-boundary safe for short ones)."""
    text = text.lower()
    for signal in signals:
        if len(signal) <= 4:
            if re.search(rf"\b{re.escape(signal)}\b", text):
                return True
        elif signal in text:
            return True
    return False


def _remote_allowed(target_locations: str) -> bool:
    """Remote-only roles pass only when 'Remote' is part of the target locations."""
    return "remote" in (target_locations or "").lower()


# --------------------------------------------------------------------------- #
# Structured scoring schema
# --------------------------------------------------------------------------- #
class JobMatchScore(BaseModel):
    match_score: int = Field(ge=0, le=100, description="Fit score from 0 to 100")
    rationale: str = Field(description="Two-sentence explanation of the fit")
    matched_skills: list[str] = Field(default_factory=list)
    missing_or_gap_skills: list[str] = Field(default_factory=list)
    job_location: str = Field(
        default="",
        description="Location as written on the posting, e.g. 'Kuala Lumpur, Malaysia'",
    )

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
                "Return ONLY a JSON array of exactly {count} search strings, each tailored "
                "to a different slice of the candidate's background.\n\n"
                "The search MUST target the target locations above. A query that could "
                "return a posting anywhere in the world is wasted, because results "
                "outside the target locations are discarded before scoring. Repeat the "
                "target location as a quoted phrase in EVERY query.\n\n"
                "Boards to use, in priority order: {boards}. The first boards listed are "
                "job boards that carry postings in the target locations; later entries are "
                "global boards that mostly return postings in other countries, so use those "
                "only if the target locations are themselves global (for example 'Remote').\n\n"
                "Rules:\n"
                "1. Every query combines one specific technical job title with the target "
                'location, e.g. "Senior Python Engineer" "Kuala Lumpur".\n'
                "2. Pin each query to one board with a site: operator. Cover as many of the "
                "location-appropriate boards as the query count allows.\n"
                "3. Aim at a single role title, never a category, so results are postings "
                "rather than listing pages.\n"
                "4. Use recent-active phrasing such as 'hiring 2026'.\n"
                "5. Never include 'remote worldwide' or 'work from anywhere' phrasing unless "
                "the target locations explicitly say Remote.\n"
                "6. No commentary, no markdown fences - raw JSON array only."
            ).format(
                roles=roles,
                locations=locations,
                resume=resume[:_RESUME_LIMIT] or "(not provided)",
                boards=", ".join(_JOB_BOARD_SITES),
                count=_QUERY_COUNT,
            )
        ),
    ]

    try:
        response = _message_text(_generate(cfg, prompt))
        queries = _extract_json(response)
        if not isinstance(queries, list):
            raise ValueError("model did not return a list")
        queries = [str(q).strip() for q in queries if str(q).strip()][:_QUERY_COUNT]
    except Exception as exc:
        logger.warning("Planner fell back to deterministic queries: %s", exc)
        queries = []

    if len(queries) < _QUERY_COUNT:
        for fallback in _fallback_queries(roles, locations):
            if fallback not in queries:
                queries.append(fallback)
            if len(queries) == _QUERY_COUNT:
                break

    logger.info("Planner produced %d queries", len(queries))
    for query in queries[:_QUERY_COUNT]:
        logger.info("  query: %s", query)
    state["queries"] = queries[:_QUERY_COUNT]
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
    # Total Tavily results stay bounded regardless of query count, while each query
    # still pulls enough rows that a real posting survives the category-page cull on
    # a narrow board (one JobStreet country site returns mostly category pages).
    max_results = max(5, cfg.max_results_per_query, 30 // max(1, len(state["queries"])))

    locations = state.get("target_locations") or cfg.target_locations
    signals = _location_signals(locations)
    remote_ok = _remote_allowed(locations)

    seen: dict[str, dict] = {}
    seen_titles: set[str] = set()
    skipped: dict[str, int] = {
        "not_a_posting": 0,
        "out_of_area": 0,
        "duplicate_title": 0,
    }
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
            title = (result.get("title") or "").strip() or "Untitled role"
            snippet = (
                result.get("content") or result.get("raw_content") or ""
            ).strip()
            if not _looks_like_posting(url, title, snippet):
                skipped["not_a_posting"] += 1
                logger.info("Skipping non-posting result %r (%s)", title, url)
                continue
            if signals:
                haystack = f"{title} {snippet} {url}".lower()
                located = _mentions_location(haystack, signals)
                remote_match = remote_ok and bool(re.search(r"\bremote\b", haystack))
                if not located and not remote_match:
                    skipped["out_of_area"] += 1
                    logger.info("Skipping out-of-area result %r (%s)", title, url)
                    continue
            title_key = _title_key(title)
            if title_key in seen_titles:
                skipped["duplicate_title"] += 1
                continue
            seen_titles.add(title_key)
            seen[url] = {
                "title": title,
                "url": url,
                "snippet": snippet[:_SNIPPET_LIMIT],
                "source": _source_from_url(url),
            }

    state["raw_jobs"] = list(seen.values())
    if failures and failures == len(state["queries"]):
        state["error"] = f"searcher: all {failures} Tavily queries failed"
        logger.error("%s", state["error"])
    logger.info(
        "Searcher collected %d unique postings (skipped: %s)",
        len(state["raw_jobs"]),
        ", ".join(f"{k}={v}" for k, v in skipped.items() if v) or "none",
    )
    return state


# --------------------------------------------------------------------------- #
# Node 3: matcher / scorer
# --------------------------------------------------------------------------- #
def _scoring_messages(resume: str, jobs: list[dict], locations: str) -> list:
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
                "Accepted locations for this search: {locations}\n\n"
                "For EVERY job id, return ONLY a JSON array (no markdown fences) of objects with:\n"
                '  "id": the original integer id,\n'
                '  "match_score": integer 0-100,\n'
                '  "rationale": exactly 2 sentences explaining the fit,\n'
                '  "matched_skills": array of skills from the resume that the posting requires,\n'
                '  "missing_or_gap_skills": array of required skills the resume lacks,\n'
                '  "job_location": the posting\'s location as written (e.g. "Kuala Lumpur, '
                'Malaysia"), or "" if none is stated.\n'
                "Include every id exactly once.\n"
                "Location rule: a posting must be based in an accepted location to score "
                "well. Score anything based elsewhere below 50, and score a fully remote "
                "role outside the accepted locations below 50 too."
            ).format(
                resume=resume[:_RESUME_LIMIT] or "(no resume provided)",
                jobs=json.dumps(catalogue, ensure_ascii=False),
                locations=locations or "(any)",
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


# --------------------------------------------------------------------------- #
# Keyword-heuristic scorer -- last-resort scoring when Gemini is unavailable
# (daily quota spent, all models 429). Keeps the digest flowing with direct
# posting URLs instead of failing the run.
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    """
    the and for you our with are will this that have from your who what when where
    about must can all not but they their them been more than then also work working
    role team skills experience years job position company ability strong well other
    some any per via within including etc may should would required requirements
    candidate ideal apply benefits benefit salary please join looking ideal youll
    across using use used help make many like new one two well get set key
    """
)


def _keyword_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9+#]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _heuristic_score(job: dict, resume_text: str) -> JobMatchScore:
    """Score by keyword overlap between the resume and the posting text.

    Calibrated so strong overlaps land in the high scores and generic-word-only
    overlap stays far below the match threshold. job_location stays empty: the
    searcher already filtered postings by location, and the matcher's location
    pass is skipped when it is empty.
    """
    resume_kw = _keyword_tokens(resume_text)
    job_kw = _keyword_tokens(f"{job.get('title', '')} {job.get('snippet', '')}")
    matched = sorted(resume_kw & job_kw)
    if resume_kw and job_kw:
        ratio = len(matched) / math.sqrt(len(resume_kw) * len(job_kw))
        score = round(min(95.0, ratio * 380.0))
    else:
        score = 0
    return JobMatchScore(
        match_score=score,
        rationale=(
            "Keyword heuristic (Gemini unavailable): "
            f"{len(matched)} resume keywords appear in this posting, "
            "so the score is overlap-based rather than a full fit analysis."
        ),
        matched_skills=matched[:10],
        missing_or_gap_skills=[],
        job_location="",
    )


# --------------------------------------------------------------------------- #
# Node 3: matcher / scorer (continues)
# --------------------------------------------------------------------------- #
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
    locations = state.get("target_locations") or cfg.target_locations
    signals = _location_signals(locations)
    remote_ok = _remote_allowed(locations)

    scores: dict[int, JobMatchScore] = {}
    try:
        response = _message_text(
            _generate(
                cfg, _scoring_messages(state.get("resume_text", ""), jobs, locations)
            )
        )
        scores = _parse_scores(response, expected_ids)
        if len(scores) < len(jobs):
            raise ValueError(f"batch scored {len(scores)}/{len(jobs)} jobs")
    except Exception as exc:
        if _is_quota_exhausted(exc):
            logger.warning("Batch scoring skipped, Gemini quota exhausted: %s", exc)
            scores = {}
        else:
            logger.warning("Batch scoring unavailable (%s); retrying per job", exc)
            scores = {}

    stop_per_job = False
    for index, job in enumerate(jobs):
        if index in scores:
            continue
        if stop_per_job:
            break
        try:
            response = _message_text(
                _generate(
                    cfg,
                    _scoring_messages(state.get("resume_text", ""), [job], locations),
                )
            )
            individual = _parse_scores(response, {index})
            if index in individual:
                scores[index] = individual[index]
            else:
                logger.warning("No score returned for %r", job.get("title"))
        except Exception as exc:
            logger.warning("Scoring failed for %r: %s", job.get("title"), exc)
            if _is_quota_exhausted(exc):
                # More calls would 429 for the rest of the day; fall through to
                # the heuristic scorer for the remaining postings.
                stop_per_job = True

    heuristic_count = 0
    if len(scores) < len(jobs):
        for index, job in enumerate(jobs):
            if index not in scores:
                scores[index] = _heuristic_score(job, state.get("resume_text", ""))
                heuristic_count += 1
        logger.warning(
            "Gemini unavailable; %d/%d postings scored by keyword heuristic",
            heuristic_count,
            len(jobs),
        )

    matches: list[dict] = []
    dropped_low, dropped_area = 0, 0
    for index, job in enumerate(jobs):
        score = scores.get(index)
        if score is None:
            continue
        if score.match_score < cfg.match_threshold:
            dropped_low += 1
            logger.debug(
                "Below threshold: %d < %d -- %r",
                score.match_score,
                cfg.match_threshold,
                job.get("title"),
            )
            continue
        if signals and score.job_location:
            location_text = score.job_location.lower()
            located = _mentions_location(location_text, signals)
            remote_match = remote_ok and bool(re.search(r"\bremote\b", location_text))
            if not located and not remote_match:
                dropped_area += 1
                logger.debug(
                    "Dropping %r: location %r outside %s",
                    job.get("title"),
                    score.job_location,
                    locations,
                )
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
        "Matcher kept %d/%d postings at or above %d%% (below=%d, out_of_area=%d)",
        len(matches),
        len(jobs),
        cfg.match_threshold,
        dropped_low,
        dropped_area,
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
