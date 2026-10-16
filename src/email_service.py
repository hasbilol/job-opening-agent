"""HTML email rendering and SMTP dispatch for job matches."""

from __future__ import annotations

import html
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse

from .config import Settings

logger = logging.getLogger(__name__)

_BADGE_COLORS = {
    "high": ("#16a34a", "#ffffff"),   # >= 85
    "medium": ("#ca8a04", "#ffffff"),  # 70 - 84
    "low": ("#6b7280", "#ffffff"),
}


def _score_bucket(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "medium"
    return "low"


def _badge(score: int) -> str:
    bg, fg = _BADGE_COLORS[_score_bucket(score)]
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'font-weight:700;font-size:13px;padding:3px 10px;border-radius:9999px;">'
        f"{int(score)}%</span>"
    )


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def _skill_list(skills: list[str]) -> str:
    if not skills:
        return '<span style="color:#9ca3af;">—</span>'
    chips = []
    for skill in skills:
        chips.append(
            '<span style="display:inline-block;background:#eef2ff;color:#3730a3;'
            'font-size:12px;padding:2px 8px;border-radius:6px;margin:0 4px 4px 0;">'
            f"{_esc(skill)}</span>"
        )
    return "".join(chips)


def _degraded_note() -> str:
    return (
        '<div style="padding:12px 32px;background:#fef3c7;border-bottom:1px solid #fcd34d;'
        'font-size:13px;color:#92400e;">'
        "Gemini's daily quota was exhausted today, so scores come from a keyword-overlap "
        "heuristic rather than a full fit analysis. Tomorrow's digest will be back to "
        "LLM scoring."
        "</div>"
    )


def render_matches_email(
    matches: list[dict], *, date_label: str, roles: str, degraded: bool = False
) -> str:
    """Render scored matches into a modern, self-contained HTML document."""
    rows = []
    for job in matches:
        score = int(job.get("match_score", 0))
        title = _esc(job.get("title", "Untitled role"))
        url = _esc(job.get("url", ""))
        rationale = _esc(job.get("rationale", ""))
        matched = job.get("key_skills_matched") or job.get("matched_skills") or []
        missing = job.get("missing_skills") or job.get("missing_or_gap_skills") or []

        title_cell = (
            f'<a href="{url}" style="color:#111827;font-weight:600;text-decoration:none;">'
            f"{title}</a>"
            if url
            else f'<span style="color:#111827;font-weight:600;">{title}</span>'
        )

        link_cell = ""
        if url:
            try:
                domain = urlparse(url).netloc.replace("www.", "")
            except ValueError:
                domain = ""
            link_cell = (
                '<div style="margin-top:10px;">'
                f'<a href="{url}" style="display:inline-block;background:#4f46e5;'
                'color:#ffffff;font-size:13px;font-weight:600;padding:8px 16px;'
                'border-radius:8px;text-decoration:none;">View posting &rarr;</a>'
                + (
                    f'<span style="margin-left:10px;font-size:12px;color:#9ca3af;">'
                    f"via {_esc(domain)}</span>"
                    if domain
                    else ""
                )
                + "</div>"
            )

        rows.append(
            "<tr>"
            f'<td style="padding:16px;border-bottom:1px solid #e5e7eb;vertical-align:top;width:90px;">'
            f"{_badge(score)}</td>"
            '<td style="padding:16px;border-bottom:1px solid #e5e7eb;vertical-align:top;">'
            f"{title_cell}"
            f'<div style="margin-top:6px;font-size:14px;color:#4b5563;line-height:1.5;">{rationale}</div>'
            f"{link_cell}"
            '<div style="margin-top:10px;"><span style="font-size:11px;font-weight:700;'
            'color:#16a34a;letter-spacing:0.5px;">MATCHED</span><br>'
            f"{_skill_list(list(matched))}</div>"
            '<div style="margin-top:6px;"><span style="font-size:11px;font-weight:700;'
            'color:#dc2626;letter-spacing:0.5px;">GAPS</span><br>'
            f"{_skill_list(list(missing))}</div>"
            "</td>"
            "</tr>"
        )

    body_rows = "".join(rows) if rows else (
        "<tr><td style=\"padding:32px;text-align:center;color:#6b7280;\">"
        "No high-confidence roles were found today.</td></tr>"
    )

    count = len(matches)
    summary = (
        f"{count} high-confidence role{'s' if count != 1 else ''} matched your profile"
        if count
        else "No roles cleared the match threshold today"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:24px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:680px;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <tr>
          <td style="background:linear-gradient(135deg,#4f46e5,#7c3aed);padding:28px 32px;">
            <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#c7d2fe;font-weight:700;">Daily Job Digest</div>
            <div style="font-size:24px;font-weight:700;color:#ffffff;margin-top:6px;">{date_label}</div>
            <div style="font-size:14px;color:#e0e7ff;margin-top:8px;">Targeting {_esc(roles)}</div>
          </td>
        </tr>
        <tr>
          <td style="padding:18px 32px;background:#f9fafb;border-bottom:1px solid #e5e7eb;font-size:14px;color:#374151;">
            <strong>{_esc(summary)}</strong> &middot; scores are 0&ndash;100 with 70 as the cut-off.
          </td>
        </tr>
        {_degraded_note() if degraded else ""}
        <tr><td style="padding:0 32px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">{body_rows}</table>
        </td></tr>
        <tr>
          <td style="padding:20px 32px;background:#f9fafb;border-top:1px solid #e5e7eb;font-size:12px;color:#9ca3af;">
            Generated automatically by the LangGraph job-search agent.
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def render_empty_email(date_label: str, roles: str, degraded: bool = False) -> str:
    """Render the 'nothing found today' notification."""
    if degraded:
        body = (
            "No roles cleared the match threshold today for "
            f"<strong>{_esc(roles)}</strong>. Gemini's daily quota was exhausted, "
            "so postings were scored with a keyword-overlap heuristic instead of the "
            "full fit analysis &mdash; strong matches may have been missed. "
            "Tomorrow's digest will be back to LLM scoring."
        )
    else:
        body = (
            "No high-confidence roles were found today for "
            f"<strong>{_esc(roles)}</strong>. "
            "The agent is resuming normal scanning &mdash; check back tomorrow."
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:24px 12px;">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border-radius:12px;overflow:hidden;">
        <tr><td style="background:#111827;padding:24px 32px;">
          <div style="font-size:20px;font-weight:700;color:#ffffff;">Daily Job Digest</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:4px;">{date_label}</div>
        </td></tr>
        <tr><td style="padding:28px 32px;font-size:15px;line-height:1.6;color:#374151;">
          {body}
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_email(settings: Settings, subject: str, html_body: str, *, date_label: str) -> str:
    """Send ``html_body`` over SMTP_SSL and return a status string.

    Raises:
        smtplib.SMTPException: On any transport-level failure.
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.email_sender
    message["To"] = settings.email_recipient
    message.attach(
        MIMEText(
            f"<p>{_esc(subject)} &mdash; {_esc(date_label)}</p>",
            "plain",
            "utf-8",
        )
    )
    message.attach(MIMEText(html_body, "html", "utf-8"))

    logger.info(
        "Connecting to %s:%s as %s -> %s",
        settings.smtp_host,
        settings.smtp_port,
        settings.email_sender,
        settings.email_recipient,
    )
    with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        server.login(settings.email_sender, settings.email_app_password)
        server.sendmail(
            settings.email_sender,
            [settings.email_recipient],
            message.as_string(),
        )

    status = f"sent to {settings.email_recipient}"
    logger.info("Email %s", status)
    return status
