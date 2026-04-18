"""Local voice assistant.

Pipeline: mic → Silero VAD → Whisper STT → Ollama (llama3.1:8b) → Piper TTS → speakers.
All local — no cloud APIs.
"""

import asyncio
import ctypes
import os
import site
import sys
from pathlib import Path


def _preload_nvidia_libs() -> None:
    """Preload CUDA libs from the nvidia-*-cu12 pip wheels so faster-whisper can find them.

    ctranslate2 (under faster-whisper) dlopen's cuBLAS/cuDNN via the loader. The
    nvidia pip wheels drop their .so files in site-packages/nvidia/*/lib/, which is
    not on LD_LIBRARY_PATH. RTLD_GLOBAL preloading makes them visible to later
    dlopen calls. Order matters: dependencies before dependents.
    """
    priority_order = ["cuda_runtime", "cublas", "cuda_nvrtc", "cudnn"]
    search_dirs: list[Path] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pkg in priority_order:
            search_dirs.extend(Path(sp).glob(f"nvidia/{pkg}/lib"))

    for d in search_dirs:
        for lib in sorted(d.glob("*.so*")):
            # Skip ".alt." variants — those are dispatch shims not meant for direct preload.
            if ".alt." in lib.name:
                continue
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_nvidia_libs()

import pyaudio  # noqa: E402
from loguru import logger  # noqa: E402

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model as WhisperModel
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

MODELS_DIR = Path(__file__).parent / "models" / "piper"
PIPER_VOICE = "en_US-lessac-medium"

SYSTEM_PROMPT = (
    "You are a friendly local voice assistant running entirely on-device. "
    "Reply in one or two short sentences — your answers will be spoken aloud."
)


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


def log_audio_devices() -> None:
    pa = pyaudio.PyAudio()
    logger.info("--- audio devices ---")
    try:
        default_in = pa.get_default_input_device_info()
        default_out = pa.get_default_output_device_info()
        logger.info(f"default input : [{default_in['index']}] {default_in['name']}")
        logger.info(f"default output: [{default_out['index']}] {default_out['name']}")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            tag_in = "in " if info["maxInputChannels"] > 0 else "   "
            tag_out = "out" if info["maxOutputChannels"] > 0 else "   "
            logger.info(f"  [{i}] {tag_in} {tag_out}  {info['name']}")
    finally:
        pa.terminate()
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
