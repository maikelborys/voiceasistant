"""Global defaults. Env vars override.

Centralises values that would otherwise be hardcoded in bot.py, runner.py,
pipeline.py, and the transport modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- LLM ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))

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
