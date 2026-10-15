"""LangGraph state schema for the daily job-search agent."""

from __future__ import annotations

from typing import TypedDict


class JobHunterState(TypedDict, total=False):
    """State passed between graph nodes.

    Attributes:
        resume_text: Full text of the candidate's resume/profile.
        target_roles: Comma-separated roles the candidate is hunting for.
        target_locations: Comma-separated target cities/countries/remote flags.
        queries: Search strings produced by the planner node.
        raw_jobs: Deduplicated scraped results with keys
            ``title``, ``url``, ``snippet``, ``source``.
        matches: Scored jobs with keys ``title``, ``url``, ``match_score``,
            ``rationale``, ``key_skills_matched``, ``missing_skills``.
        email_status: Human-readable outcome of the dispatch node.
        error: Populated when a node fails so downstream nodes can no-op.
    """

    resume_text: str
    target_roles: str
    target_locations: str
    queries: list[str]
    raw_jobs: list[dict]
    matches: list[dict]
    email_status: str
    error: str
