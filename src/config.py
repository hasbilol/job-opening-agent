"""Environment loading and validation for the job-search agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Load .env from the repo root if present; real system env vars take precedence.
load_dotenv(override=False)

_REQUIRED_KEYS = (
    "GEMINI_API_KEY",
    "TAVILY_API_KEY",
    "EMAIL_SENDER",
    "EMAIL_APP_PASSWORD",
    "EMAIL_RECIPIENT",
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _get(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_int(key: str, default: int) -> int:
    raw = _get(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration."""

    gemini_api_key: str
    tavily_api_key: str
    email_sender: str
    email_app_password: str
    email_recipient: str

    gemini_model: str = "gemini-1.5-flash"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465

    target_roles: str = "Backend Engineer, Machine Learning Engineer"
    target_locations: str = "Singapore, Remote"
    match_threshold: int = 70
    max_results_per_query: int = 5

    missing: tuple[str, ...] = field(default=(), compare=False)


def load_settings(*, strict: bool = True) -> Settings:
    """Load settings from the environment.

    Args:
        strict: When True, raise ``ConfigError`` if any required secret is absent.
            When False, missing secrets are collected on ``Settings.missing`` so the
            CLI can report every problem at once instead of failing on the first.
    """
    values = {key: _get(key) for key in _REQUIRED_KEYS}
    missing = tuple(key for key, value in values.items() if not value)

    if missing and strict:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )

    settings = Settings(
        gemini_api_key=values["GEMINI_API_KEY"],
        tavily_api_key=values["TAVILY_API_KEY"],
        email_sender=values["EMAIL_SENDER"],
        email_app_password=values["EMAIL_APP_PASSWORD"],
        email_recipient=values["EMAIL_RECIPIENT"],
        gemini_model=_get("GEMINI_MODEL", "gemini-1.5-flash"),
        smtp_host=_get("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=_get_int("SMTP_PORT", 465),
        target_roles=_get("TARGET_ROLES", "Backend Engineer, Machine Learning Engineer"),
        target_locations=_get("TARGET_LOCATIONS", "Singapore, Remote"),
        match_threshold=_get_int("MATCH_THRESHOLD", 70),
        max_results_per_query=max(1, _get_int("MAX_RESULTS_PER_QUERY", 5)),
        missing=missing,
    )

    if not 0 <= settings.match_threshold <= 100:
        raise ConfigError("MATCH_THRESHOLD must be between 0 and 100.")

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached, strictly-validated settings (loaded lazily, never at import time)."""
    return load_settings(strict=True)
