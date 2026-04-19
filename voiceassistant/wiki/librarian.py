"""Daily-log append. Non-LLM — structured write per turn.

MVP is append-only. Upgrade path (deferred Phase 8.3): replace with an
LLM-driven librarian that diffs pages against `schema.md` and emits
targeted edits instead of just appending.
"""

from __future__ import annotations

from datetime import datetime

from voiceassistant.session import SessionContext
from voiceassistant.wiki.store import append_page


def append_daily_log(session: SessionContext, user_text: str, bot_text: str) -> None:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    block = (
        f"## {time_str} — {session.user_id} on {session.device_id} "
        f"(persona: {session.persona_id})\n"
        f"**user:** {user_text}\n"
        f"**bot:** {bot_text}\n\n"
    )
    append_page(f"daily/{date_str}.md", block)

    def _clip(s: str, n: int = 80) -> str:
        s = s.replace("\n", " ").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    log_line = (
        f"{date_str} {time_str} [{session.short_id}] "
        f"{session.user_id}@{session.device_id}/{session.persona_id} · "
        f"{_clip(user_text)} → {_clip(bot_text)}\n"
    )
    append_page("log.md", log_line)
