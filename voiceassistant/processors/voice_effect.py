"""Character-voice post-processor — pitch-shift TTS audio by N semitones.

Sits after PiperTTSService in the pipeline. Piper produces natural human
voices; this bends them toward cartoon characters (squeaky doll, deep
dragon) by resampling each TTSAudioRawFrame.

Positive semitones = higher pitch; negative = lower. Duration changes
inversely (up-shift shortens audio, down-shift lengthens). Per-chunk
polyphase filter state causes minor artifacts at chunk boundaries —
acceptable for a toy; not broadcast-grade.

Pure CPU, scipy-based. No GPU contention with the LLM.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from scipy.signal import resample_poly


class PitchShift(FrameProcessor):
    def __init__(self, semitones: float):
        super().__init__()
        self._semitones = float(semitones)
        ratio = 2.0 ** (self._semitones / 12.0)
        # resample_poly(x, up, down) -> len(out) ≈ len(x) * up / down.
        # Higher pitch = fewer samples at the same sample_rate, so up/down = 1/ratio.
        denom = 1000
        self._down = max(1, int(round(denom * ratio)))
        self._up = denom
        g = gcd(self._up, self._down)
        self._up //= g
        self._down //= g
        self._enabled = self._up != self._down

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._enabled and isinstance(frame, TTSAudioRawFrame):
            try:
                audio = np.frombuffer(frame.audio, dtype=np.int16)
                if audio.size:
                    shifted = resample_poly(audio.astype(np.float32), self._up, self._down)
                    shifted = np.clip(shifted, -32768, 32767).astype(np.int16)
                    frame.audio = shifted.tobytes()
                    frame.num_frames = int(shifted.size)
            except Exception as e:
                logger.warning(f"PitchShift: passthrough after error ({e})")
        await self.push_frame(frame, direction)
