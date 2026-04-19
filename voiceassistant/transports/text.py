"""Text stdin/stdout transport — no STT, no TTS, no VAD.

Debug superpower for Phase 8: skip the ~20s model-loading tax of the
local_audio pipeline and iterate on prompts/memory/tools at ~1s startup.

StdinTextInput reads lines from sys.stdin in a background task. Each
non-empty line becomes a UserStarted → Transcription → UserStopped
burst, matching what STT would produce. EOF pushes EndFrame.

StdoutTextOutput writes streaming LLM tokens to stdout live, then a
newline + prompt char after each full response. Speech-lifecycle frames
are observed but not rendered (SpeechEventLogger still logs them).
"""

from __future__ import annotations

import sys

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

from voiceassistant.session import SessionContext
from voiceassistant.transports import TransportBundle

_PROMPT = "> "


class StdinTextInput(FrameProcessor):
    """Reads sys.stdin line-by-line and injects TranscriptionFrames downstream."""

    def __init__(self, session: SessionContext) -> None:
        super().__init__()
        self._session = session
        self._reader_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._reader_task is None:
            sys.stdout.write(_PROMPT)
            sys.stdout.flush()
            self._reader_task = self.create_task(self._read_stdin(), name="stdin-reader")
        await self.push_frame(frame, direction)

    async def _read_stdin(self) -> None:
        loop = self.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if line == "":
                logger.info("stdin EOF — shutting down pipeline")
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                return
            text = line.rstrip("\n").strip()
            if not text:
                sys.stdout.write(_PROMPT)
                sys.stdout.flush()
                continue
            await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id=self._session.user_id,
                    timestamp=time_now_iso8601(),
                    finalized=True,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


class StdoutTextOutput(FrameProcessor):
    """Streams LLMTextFrames to stdout; newline + prompt on response end."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            sys.stdout.write(frame.text)
            sys.stdout.flush()
        elif isinstance(frame, LLMFullResponseEndFrame):
            sys.stdout.write("\n" + _PROMPT)
            sys.stdout.flush()
        await self.push_frame(frame, direction)


def build_text_bundle(session: SessionContext) -> TransportBundle:
    return TransportBundle(
        input=StdinTextInput(session),
        output=StdoutTextOutput(),
        needs_stt=False,
        needs_tts=False,
        needs_vad=False,
        wants_user_mute=False,
    )
