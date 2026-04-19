"""Local audio transport bundle — wraps pipecat LocalAudioTransport.

Logs the device list at construction time so step-3 smoke tests match
step-2 output exactly. STT/TTS/VAD/user-mute are all enabled.
"""

from __future__ import annotations

from loguru import logger
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from voiceassistant.audio_devices import (
    default_input_device,
    default_output_device,
    list_devices,
)
from voiceassistant.session import SessionContext
from voiceassistant.transports import TransportBundle


def _log_audio_devices() -> None:
    din = default_input_device()
    dout = default_output_device()
    logger.info("--- audio devices ---")
    logger.info(f"default input : [{din.index}] {din.name}")
    logger.info(f"default output: [{dout.index}] {dout.name}")
    for d in list_devices():
        tag_in = "in " if d.is_input else "   "
        tag_out = "out" if d.is_output else "   "
        logger.info(f"  [{d.index}] {tag_in} {tag_out}  {d.name}")
    logger.info("---------------------")


def build_local_audio_bundle(session: SessionContext) -> TransportBundle:
    _log_audio_devices()
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    return TransportBundle(
        input=transport.input(),
        output=transport.output(),
        needs_stt=True,
        needs_tts=True,
        needs_vad=True,
        wants_user_mute=True,
    )
