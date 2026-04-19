"""Transport factory + bundle abstraction.

A TransportBundle is a pair of FrameProcessors (input + output) plus flags
that tell the pipeline whether to splice STT/TTS/VAD/user-mute around them.
The same `build_pipeline()` then works for local audio, text stdin/stdout,
and (later) WebSocket without branching.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.processors.frame_processor import FrameProcessor

from voiceassistant.session import SessionContext


@dataclass
class TransportBundle:
    input: FrameProcessor
    output: FrameProcessor
    needs_stt: bool
    needs_tts: bool
    needs_vad: bool
    wants_user_mute: bool
    sample_rate: int = 16000


def make_transport(session: SessionContext) -> TransportBundle:
    if session.transport_kind == "local_audio":
        from voiceassistant.transports.local_audio import build_local_audio_bundle

        return build_local_audio_bundle(session)
    if session.transport_kind == "text":
        from voiceassistant.transports.text import build_text_bundle

        return build_text_bundle(session)
    if session.transport_kind == "websocket":
        raise NotImplementedError(
            "WebSocket transport scheduled for Phase 9 (ESP32 frontend)."
        )
    raise ValueError(f"Unknown transport: {session.transport_kind}")
