"""Pipeline assembly — transport-agnostic.

Given a SessionContext and a TransportBundle, wires up:

    bundle.input
    → SpeechEventLogger
    → [WhisperSTT if bundle.needs_stt]
    → aggregators.user()            (VAD + user-mute gated by bundle flags)
    → OLLamaLLMService
    → [PiperTTS if bundle.needs_tts]
    → bundle.output
    → aggregators.assistant()

Persona + wiki-retrieval + librarian splice in during steps 4, 6, 7.
"""

from __future__ import annotations

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
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
from pipecat.transcriptions.language import Language
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from voiceassistant import config
from voiceassistant.personas import Persona
from voiceassistant.processors.speech_logger import SpeechEventLogger
from voiceassistant.processors.vector_retrieval import VectorRetrieval
from voiceassistant.processors.wiki_librarian import WikiLibrarian
from voiceassistant.session import SessionContext
from voiceassistant.transports import TransportBundle


def build_pipeline(
    session: SessionContext, bundle: TransportBundle, persona: Persona
) -> PipelineTask:
    stages = [bundle.input, SpeechEventLogger()]

    if bundle.needs_stt:
        whisper_model_name = persona.whisper_model or config.WHISPER_MODEL
        stt_settings_kwargs: dict = {
            "model": getattr(WhisperModel, whisper_model_name).value,
        }
        if persona.language:
            stt_settings_kwargs["language"] = Language(persona.language)
        stages.append(
            WhisperSTTService(
                settings=WhisperSTTService.Settings(**stt_settings_kwargs),
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        )

    context = LLMContext(messages=[{"role": "system", "content": persona.system_prompt}])
    user_params_kwargs: dict = {}
    if bundle.needs_vad:
        user_params_kwargs["vad_analyzer"] = SileroVADAnalyzer()
    if bundle.wants_user_mute:
        user_params_kwargs["user_mute_strategies"] = [AlwaysUserMuteStrategy()]
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(**user_params_kwargs),
    )
    stages.append(aggregators.user())
    stages.append(
        VectorRetrieval(
            session=session,
            context=context,
            base_system_prompt=persona.system_prompt,
            budget_chars=config.WIKI_INJECT_BUDGET_CHARS,
        )
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=config.OLLAMA_MODEL, temperature=config.OLLAMA_TEMPERATURE
        ),
    )
    stages.append(llm)

    if bundle.needs_tts:
        stages.append(
            PiperTTSService(
                settings=PiperTTSService.Settings(voice=persona.piper_voice),
                download_dir=config.MODELS_DIR,
                use_cuda=False,
            )
        )

    stages.append(bundle.output)
    stages.append(aggregators.assistant())
    stages.append(WikiLibrarian(session=session, context=context))

    pipeline = Pipeline(stages)
    return PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
