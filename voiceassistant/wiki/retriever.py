"""Pick which wiki pages to inject into the LLM system prompt.

MVP set: active persona + active device + active user + last block of
today's daily log. Missing pages are silently skipped. Keyword-based
suggestion from `index.md` is a later phase.
"""

from __future__ import annotations

from datetime import date

from voiceassistant.session import SessionContext
from voiceassistant.wiki.store import read_page


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
    if daily is not None:
        blocks = [b for b in daily.strip().split("\n## ") if b.strip()]
        if blocks:
            last = blocks[-1]
            if not last.startswith("## "):
                last = "## " + last
            out.append((f"daily/{today}.md (most recent turn)", last.strip()))
    return out
