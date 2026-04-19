"""Global defaults. Env vars override.

Centralises values that would otherwise be hardcoded in bot.py, runner.py,
pipeline.py, and the transport modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- LLM ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Custom tag built from models/ollama/llama3.1-ctx4k.Modelfile.
# Stock llama3.1:8b defaults to num_ctx=2048, which truncated older turns and
# caused mid-story forgetting. 4k is the largest KV cache we can afford on
# an 8 GB card alongside Whisper turbo + nomic-embed without OOM risk.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-ctx4k")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))

# OpenRouter (OpenAI-compatible) — cloud opt-in via --llm openrouter/<model>.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
# Default LLM spec if --llm not supplied. "local" resolves to OLLAMA_MODEL.
LLM_DEFAULT_SPEC = os.environ.get("LLM_DEFAULT_SPEC", "local")

# --- Embeddings (nomic-embed-text via Ollama) ---
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# --- STT (faster-whisper) ---
# Must be a name from pipecat.services.whisper.stt.Model enum.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "DISTIL_MEDIUM_EN")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

# --- VAD (Silero) ---
# stop_secs of silence before VAD calls a turn done. Pipecat's default is
# 0.2 which cuts speakers mid-pause; 0.8 forgives natural thinking pauses
# without feeling laggy. start_secs stays at 0.2 to still catch fast starts.
VAD_STOP_SECS = float(os.environ.get("VAD_STOP_SECS", "0.8"))
VAD_START_SECS = float(os.environ.get("VAD_START_SECS", "0.2"))
VAD_CONFIDENCE = float(os.environ.get("VAD_CONFIDENCE", "0.7"))
VAD_MIN_VOLUME = float(os.environ.get("VAD_MIN_VOLUME", "0.6"))

# --- TTS (Piper) ---
PIPER_VOICE_DEFAULT = os.environ.get("PIPER_VOICE_DEFAULT", "en_US-lessac-medium")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "piper"

# --- Wiki retrieval ---
WIKI_INJECT_BUDGET_CHARS = int(os.environ.get("WIKI_INJECT_BUDGET_CHARS", "8000"))

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
