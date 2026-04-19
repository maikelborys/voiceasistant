"""CLI entrypoint.

Parses --transport/--user/--device/--persona, builds a SessionContext,
dispatches to the matching transport runner. Currently only local_audio is
wired; text transport lands in step 5 and websocket in Phase 9 (ESP32).

The real pipeline assembly moves into voiceassistant.pipeline in step 3. For
step 2 we inline a v1-equivalent local-audio pipeline here so the entrypoint
boundary is proven working before the TransportBundle refactor.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from voiceassistant import config
from voiceassistant.session import SessionContext, TransportKind


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voiceassistant",
        description="PipecatAssistant — modular local voice agent.",
    )
    parser.add_argument(
        "--transport",
        choices=["text", "local_audio", "websocket"],
        default="local_audio",
        help="Input/output frontend (default: local_audio)",
    )
    parser.add_argument(
        "--user",
        default="maikel",
        help="User ID — picks wiki/people/<user>.md (default: maikel)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device ID — defaults to 'stdin' for text, 'laptop' for local_audio",
    )
    parser.add_argument(
        "--persona",
        default="default",
        help="Persona ID — picks voice + system prompt (default: default)",
    )
    return parser.parse_args()


def _default_device_for(transport: TransportKind) -> str:
    return {"text": "stdin", "local_audio": "laptop", "websocket": "websocket"}[transport]


async def _run_local_audio(session: SessionContext) -> None:
    """v1-equivalent pipeline. Step 3 moves this into voiceassistant.pipeline."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.ollama.llm import OLLamaLLMService
    from pipecat.services.piper.tts import PiperTTSService
    from pipecat.services.whisper.stt import Model as WhisperModel
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.turns.user_mute import AlwaysUserMuteStrategy

    from voiceassistant.audio_devices import (
        default_input_device,
        default_output_device,
        list_devices,
    )
    from voiceassistant.processors.speech_logger import SpeechEventLogger

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

    system_prompt = (
        "You are a friendly local voice assistant running entirely on-device. "
        "Reply in one or two short sentences — your answers will be spoken aloud."
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=getattr(WhisperModel, config.WHISPER_MODEL).value
        ),
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=config.OLLAMA_MODEL, temperature=config.OLLAMA_TEMPERATURE
        ),
    )
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.PIPER_VOICE_DEFAULT),
        download_dir=config.MODELS_DIR,
        use_cuda=False,
    )

    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            SpeechEventLogger(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    logger.info(
        f"session {session.short_id}: ready — "
        f"user={session.user_id} device={session.device_id} persona={session.persona_id}"
    )
    logger.info("Speak into your microphone.")
    await PipelineRunner().run(task)


async def async_main() -> None:
    args = _parse_args()

    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)

    device = args.device or _default_device_for(args.transport)
    session = SessionContext.new(
        transport_kind=args.transport,
        device_id=device,
        user_id=args.user,
        persona_id=args.persona,
    )
    logger.info(
        f"session {session.short_id}: starting — "
        f"transport={session.transport_kind} user={session.user_id} "
        f"device={session.device_id} persona={session.persona_id}"
    )

    if args.transport == "local_audio":
        await _run_local_audio(session)
    elif args.transport == "text":
        raise NotImplementedError("Text transport lands in step 5 of the Phase 8 plan.")
    elif args.transport == "websocket":
        raise NotImplementedError(
            "WebSocket transport scheduled for Phase 9 (ESP32 frontend)."
        )


def main() -> None:
    asyncio.run(async_main())
