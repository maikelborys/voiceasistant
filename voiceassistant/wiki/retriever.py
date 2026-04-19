"""Pick which wiki pages to inject into the LLM system prompt.

Cold-path fallback used by `VectorRetrieval` when the sqlite-vec index
is empty or embeddings are unavailable. Returns active persona +
active device + active user + today's user statements (from daily
log) as `(label, body)` pairs. Missing pages are silently skipped.
Budget enforcement lives in the processor.

Why only user statements from the daily log, not the whole log:
injecting the bot's prior responses creates a hallucination feedback
loop — a wrong answer in one session gets read as truth in the next and
repeated. User statements are the closest thing to durable facts we
have until an LLM librarian promotes them to `people/<user>.md`
(Phase 8.3). This is the raw-source-as-ground-truth shortcut.
"""

from __future__ import annotations

import re
from datetime import date

from voiceassistant.session import SessionContext
from voiceassistant.wiki.store import read_page

_USER_LINE = re.compile(r"^\*\*user:\*\*\s*(.+)$", re.MULTILINE)
_BLOCK_HEADER = re.compile(r"^## (\d{2}:\d{2}:\d{2}) — ", re.MULTILINE)


def _user_statements_from_daily(daily: str) -> str:
    """Extract only `## HH:MM:SS` + `**user:** ...` lines, drop bot responses."""
    lines: list[str] = []
    current_ts: str | None = None
    for line in daily.splitlines():
        header = _BLOCK_HEADER.match(line)
        if header:
            current_ts = header.group(1)
            continue
        user = _USER_LINE.match(line)
        if user and current_ts:
            lines.append(f"- [{current_ts}] {user.group(1).strip()}")
            current_ts = None
    return "\n".join(lines)


def pages_for_session(session: SessionContext) -> list[tuple[str, str]]:
    """Return (label, body) pairs in injection order. Missing pages skipped."""
    out: list[tuple[str, str]] = []
    for rel in (
        f"personas/{session.persona_id}.md",
        f"devices/{session.device_id}.md",
        f"people/{session.user_id}.md",
    ):
        body = read_page(rel)
        if body is not None:
            out.append((rel, body.strip()))

    today = date.today().isoformat()
    daily = read_page(f"daily/{today}.md")
    if daily is not None and daily.strip():
        user_only = _user_statements_from_daily(daily)
        if user_only:
            out.append(
                (f"daily/{today}.md (today's user statements)", user_only)
            )
    return out
