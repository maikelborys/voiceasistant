"""LLM backend factory — resolves a spec string to a Pipecat LLM service.

Spec grammar:
  local                                   -> Ollama, config.OLLAMA_MODEL
  ollama                                  -> Ollama, config.OLLAMA_MODEL
  ollama/<model>                          -> Ollama, <model> (e.g. ollama/llama3.1:8b)
  openrouter/<model>                      -> OpenRouter, <model>
                                             (e.g. openrouter/anthropic/claude-sonnet-4.5)

OpenRouter is OpenAI-compatible, so we reuse Pipecat's OpenAILLMService with a
custom base_url. Tool registration, streaming, and context aggregation all work
identically across both backends.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pipecat.services.llm_service import LLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.openai.llm import OpenAILLMService

from voiceassistant import config


@dataclass(frozen=True)
class ResolvedLLM:
    backend: str   # "ollama" | "openrouter"
    model: str


def parse_spec(spec: str) -> ResolvedLLM:
    spec = (spec or "").strip()
    if spec in ("", "local", "ollama"):
        return ResolvedLLM(backend="ollama", model=config.OLLAMA_MODEL)
    head, sep, tail = spec.partition("/")
    if not sep or not tail:
        raise ValueError(
            f"Invalid --llm spec {spec!r}. "
            "Expected 'local', 'ollama[/<model>]', or 'openrouter/<model>'."
        )
    head = head.lower()
    if head == "ollama":
        return ResolvedLLM(backend="ollama", model=tail)
    if head == "openrouter":
        return ResolvedLLM(backend="openrouter", model=tail)
    raise ValueError(
        f"Unknown LLM backend {head!r} in spec {spec!r}. "
        "Supported: 'local', 'ollama', 'openrouter'."
    )


def build_llm(spec: str, *, temperature: float | None = None) -> LLMService:
    resolved = parse_spec(spec)
    temp = config.OLLAMA_TEMPERATURE if temperature is None else temperature

    if resolved.backend == "ollama":
        logger.info(f"LLM backend: ollama model={resolved.model}")
        return OLLamaLLMService(
            settings=OLLamaLLMService.Settings(
                model=resolved.model, temperature=temp
            ),
        )

    if resolved.backend == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before using "
                "--llm openrouter/<model>."
            )
        logger.info(f"LLM backend: openrouter model={resolved.model}")
        return OpenAILLMService(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            settings=OpenAILLMService.Settings(
                model=resolved.model, temperature=temp
            ),
        )

    raise RuntimeError(f"Unreachable backend: {resolved.backend}")
