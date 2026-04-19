"""save_note tool — append a timestamped bullet to wiki/notes/<today>.md.

Lightweight user-dictated note capture. Separate from the librarian
(which curates people/ and topics/) and from the daily log (which is
a turn-by-turn transcript). Notes are the "write this down for me"
channel — passwords, quick reminders, addresses, etc.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from voiceassistant.wiki.store import append_page

NAME = "save_note"

SCHEMA = FunctionSchema(
    name=NAME,
    description=(
        "Append a short note to today's notes page. Use when the user "
        "explicitly asks to remember, note down, or write something. "
        "Notes survive across sessions and are searchable via memory_search."
    ),
    properties={
        "text": {
            "type": "string",
            "description": "The note content to persist, in the user's own words.",
        },
    },
    required=["text"],
)


async def handler(params: FunctionCallParams) -> None:
    text = str((params.arguments or {}).get("text", "")).strip()
    if not text:
        await params.result_callback({"result": "empty note, nothing saved"})
        return

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    line = f"- [{time_str}] {text}\n"
    append_page(f"notes/{date_str}.md", line)
    logger.info(f"save_note: appended to notes/{date_str}.md")
    await params.result_callback({"result": f"saved to notes/{date_str}.md"})
