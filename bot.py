"""Local voice assistant — v1 entrypoint.

Pipeline: mic → Silero VAD → Whisper STT → Ollama (llama3.1:8b) → Piper TTS → speakers.
All local — no cloud APIs.

Note: this file is a thin entrypoint. Shared helpers live in the voiceassistant/
package. Subsequent phases move pipeline assembly and transport selection into
voiceassistant.runner; for now bot.py still wires everything so v1 remains
bit-for-bit identical to the 2026-04-17 verified state.
"""

# --- CUDA preload MUST happen before any pipecat / faster-whisper import ---
from voiceassistant.preload import preload_nvidia_libs

preload_nvidia_libs()

import asyncio  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from loguru import logger  # noqa: E402

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.ollama.llm import OLLamaLLMService  # noqa: E402
from pipecat.services.piper.tts import PiperTTSService  # noqa: E402
from pipecat.services.whisper.stt import Model as WhisperModel  # noqa: E402
from pipecat.services.whisper.stt import WhisperSTTService  # noqa: E402
from pipecat.transports.local.audio import (  # noqa: E402
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy  # noqa: E402

from voiceassistant.audio_devices import (  # noqa: E402
    default_input_device,
    default_output_device,
    list_devices,
)
from voiceassistant.processors.speech_logger import SpeechEventLogger  # noqa: E402

from pathlib import Path  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models" / "piper"
PIPER_VOICE = "en_US-lessac-medium"

SYSTEM_PROMPT = (
    "You are a friendly local voice assistant running entirely on-device. "
    "Reply in one or two short sentences — your answers will be spoken aloud."
)


def log_audio_devices() -> None:
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


async def main() -> None:
    logger.remove()
    log_level = os.environ.get("LOG_LEVEL", "DEBUG")
    logger.add(sys.stderr, level=log_level)

    log_audio_devices()

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=WhisperModel.DISTIL_MEDIUM_EN.value),
        device="cuda",
        compute_type="float16",
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(model="llama3.1:8b", temperature=0.7),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=PIPER_VOICE),
        download_dir=MODELS_DIR,
        use_cuda=False,
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    logger.info("Starting voice assistant — speak into your microphone.")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
