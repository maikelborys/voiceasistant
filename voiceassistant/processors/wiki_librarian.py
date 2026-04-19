"""Pipeline-tail processor that appends each turn to the daily log.

Placed after aggregators.assistant() so the LLMContext already has the
new user + assistant messages when LLMFullResponseEndFrame arrives.
File writes run in a background task so the next turn is never blocked.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextAssistantTimestampFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voiceassistant.session import SessionContext
from voiceassistant.wiki.librarian import append_daily_log


def _content_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        ).strip()
    return str(content or "").strip()


class WikiLibrarian(FrameProcessor):
    def __init__(self, session: SessionContext, context: LLMContext) -> None:
        super().__init__()
        self._session = session
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextAssistantTimestampFrame):
            self._append_now()
        await self.push_frame(frame, direction)

    def _append_now(self) -> None:
        user_text = ""
        bot_text = ""
        for m in reversed(self._context.messages):
            role = m.get("role")
            text = _content_text(m.get("content"))
            if role == "assistant" and not bot_text:
                bot_text = text
            elif role == "user" and not user_text:
                user_text = text
            if user_text and bot_text:
                break
        if not user_text or not bot_text:
            logger.debug("WikiLibrarian: no user/assistant pair yet — skipping")
            return
        append_daily_log(self._session, user_text, bot_text)
        logger.debug(f"WikiLibrarian: appended turn for {self._session.short_id}")
