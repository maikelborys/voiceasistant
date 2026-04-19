"""Global defaults. Env vars override.

Centralises values that would otherwise be hardcoded in bot.py, runner.py,
pipeline.py, and the transport modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- LLM ---
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))

# --- STT (faster-whisper) ---
# Must be a name from pipecat.services.whisper.stt.Model enum.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "DISTIL_MEDIUM_EN")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

# --- TTS (Piper) ---
PIPER_VOICE_DEFAULT = os.environ.get("PIPER_VOICE_DEFAULT", "en_US-lessac-medium")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "piper"

# --- Wiki retrieval ---
WIKI_INJECT_BUDGET_CHARS = int(os.environ.get("WIKI_INJECT_BUDGET_CHARS", "8000"))

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
