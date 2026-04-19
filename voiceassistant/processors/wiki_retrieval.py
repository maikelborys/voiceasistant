"""Pre-LLM system-message injector.

Sits between user aggregator and LLM. On every user turn start, rebuilds
messages[0] as `<persona prompt>\\n\\n--- wiki pages ---`, giving the LLM
fresh context each turn without bloating the conversation history.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, UserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voiceassistant.session import SessionContext
from voiceassistant.wiki.retriever import pages_for_session


class WikiRetrieval(FrameProcessor):
    def __init__(
        self,
        session: SessionContext,
        context: LLMContext,
        base_system_prompt: str,
        budget_chars: int,
    ) -> None:
        super().__init__()
        self._session = session
        self._context = context
        self._base_prompt = base_system_prompt
        self._budget = budget_chars

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self._inject()
        await self.push_frame(frame, direction)

    def _inject(self) -> None:
        pages = pages_for_session(self._session)
        parts = [self._base_prompt]
        remaining = self._budget - len(self._base_prompt)
        for label, body in pages:
            block = f"\n\n--- {label} ---\n{body}"
            if len(block) > remaining:
                block = block[: max(remaining, 0)] + "\n...[truncated]"
            parts.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break
        new_system = "".join(parts)
        messages = self._context.messages
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": new_system}
        else:
            messages.insert(0, {"role": "system", "content": new_system})
        logger.debug(
            f"WikiRetrieval: injected {len(new_system)} chars from {len(pages)} pages"
        )
