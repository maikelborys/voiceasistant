"""Pipeline frame logger with direction arrows and emojis.

Dropped into the pipeline right after transport.input(). Makes user/bot/mute/
transcription events visible so debugging the pipeline is not guesswork.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SpeechEventLogger(FrameProcessor):
    """Prints clear user/bot/mute/transcription events so we can see what the pipeline hears."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        arrow = "↓" if direction == FrameDirection.DOWNSTREAM else "↑"
        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info(f"🎤 {arrow} VAD: voice detected")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info(f"🎤 {arrow} VAD: voice stopped")
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info(f"🎤 {arrow} user started speaking (turn started)")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"🎤 {arrow} user stopped speaking (turn ended)")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(f"🔊 {arrow} bot started speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(f"🔊 {arrow} bot stopped speaking")
        elif isinstance(frame, UserMuteStartedFrame):
            logger.info(f"🔇 {arrow} user muted")
        elif isinstance(frame, UserMuteStoppedFrame):
            logger.info(f"🔈 {arrow} user unmuted")
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.info(f"📝 {arrow} interim: {frame.text!r}")
        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"📝 {arrow} final:   {frame.text!r}")
        await self.push_frame(frame, direction)
