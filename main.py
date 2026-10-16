"""CLI entrypoint for the daily LangGraph job-search agent.

Usage:
    python main.py
    python main.py --resume resume.txt --roles "Backend Engineer" --locations "Singapore, Remote"
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_settings
from src.graph import build_graph, NODE_ORDER
from src.state import JobHunterState

ROOT = Path(__file__).resolve().parent
DEFAULT_RESUME = ROOT / "resume.txt"

logger = logging.getLogger("job_agent")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan active job boards and email a scored digest.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to a resume text file. Overrides the RESUME_B64/RESUME_TEXT "
            "environment secrets. Default: RESUME_B64 > RESUME_TEXT > resume.txt"
        ),
    )
    parser.add_argument(
        "--roles",
        default=None,
        help="Comma-separated target roles (default: TARGET_ROLES env var)",
    )
    parser.add_argument(
        "--locations",
        default=None,
        help="Comma-separated target locations (default: TARGET_LOCATIONS env var)",
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Print the final graph state as JSON on stdout",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def _print(message: str) -> None:
    print(message, flush=True)


def _read_resume(path: Path, *, allow_env: bool = True) -> tuple[str, str]:
    """Load the resume, preferring private env secrets over the repo file.

    Priority: ``RESUME_B64`` > ``RESUME_TEXT`` > ``path`` (returns text and a
    human-readable source label). The repo's resume.txt stays a non-sensitive
    template; a real resume is supplied via GitHub Actions secrets so it never
    lands in git history.
    """
    if allow_env:
        encoded = os.getenv("RESUME_B64", "").strip()
        if encoded:
            try:
                text = base64.b64decode(encoded).decode("utf-8").strip()
            except Exception as exc:
                logger.warning("RESUME_B64 could not be decoded (%s); ignoring it", exc)
            else:
                if text:
                    return text, "RESUME_B64 secret"
        text = os.getenv("RESUME_TEXT", "").strip()
        if text:
            return text, "RESUME_TEXT secret"

    if path.exists():
        return path.read_text(encoding="utf-8").strip(), str(path)

    logger.warning("Resume file %s not found; continuing without it", path)
    return "", str(path)


def _describe(node: str, update: dict[str, Any]) -> str:
    if node == "planner":
        queries = update.get("queries") or []
        return f"{len(queries)} queries generated"
    if node == "searcher":
        jobs = update.get("raw_jobs") or []
        sources = sorted({job.get("source", "?") for job in jobs})
        detail = f" from {', '.join(sources)}" if sources else ""
        return f"{len(jobs)} unique postings{detail}"
    if node == "matcher":
        matches = update.get("matches") or []
        top = ", ".join(
            f"{m.get('title', '?')} ({m.get('match_score', 0)}%)" for m in matches[:3]
        )
        mode = update.get("scoring_mode")
        mode_note = f" | scoring: {mode}" if mode and mode != "llm" else ""
        return (
            f"{len(matches)} matches >= threshold"
            + (f" | top: {top}" if top else "")
            + mode_note
        )
    if node == "mailer":
        return str(update.get("email_status") or "no status")
    if update.get("error"):
        return f"ERROR -> {update['error']}"
    return "done"


def run(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(strict=True)
    except ConfigError as exc:
        _print(f"[config] {exc}")
        return 2

    resume_path = args.resume or DEFAULT_RESUME
    resume_text, resume_source = _read_resume(
        resume_path, allow_env=args.resume is None
    )
    roles = args.roles or settings.target_roles
    locations = args.locations or settings.target_locations

    state: JobHunterState = {
        "resume_text": resume_text,
        "target_roles": roles,
        "target_locations": locations,
        "queries": [],
        "raw_jobs": [],
        "matches": [],
        "email_status": "",
    }

    _print("=" * 70)
    _print("  DAILY JOB SEARCH AGENT")
    _print("=" * 70)
    _print(f"  resume    : {resume_source} ({len(resume_text)} chars)")
    _print(f"  roles     : {roles}")
    _print(f"  locations : {locations}")
    _print(f"  model     : {settings.gemini_model} | threshold: {settings.match_threshold}%")
    _print("-" * 70)

    graph = build_graph(settings)
    current: dict[str, Any] = dict(state)
    total = len(NODE_ORDER)
    previous_error: str | None = None

    for step, chunk in enumerate(graph.stream(state, stream_mode="updates"), start=1):
        for node, update in chunk.items():
            if node == "__end__":
                continue
            current.update(update or {})
            index = NODE_ORDER.index(node) + 1 if node in NODE_ORDER else step
            # Only flag FAIL when this node introduced the error, not when it
            # merely inherited an earlier node's error in the shared state.
            node_error = (update or {}).get("error")
            marker = "FAIL" if node_error and node_error != previous_error else "OK  "
            previous_error = node_error or previous_error
            _print(f"[{index}/{total}] {marker} {node:<9} {_describe(node, update)}")

    _print("-" * 70)
    matches = current.get("matches") or []
    _print(f"  matches   : {len(matches)}")
    _print(f"  email     : {current.get('email_status') or 'not sent'}")
    if current.get("error"):
        _print(f"  error     : {current['error']}")
    _print("=" * 70)

    if args.dump_state:
        _print(json.dumps(current, indent=2, ensure_ascii=False, default=str))

    return 1 if current.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        _print("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
