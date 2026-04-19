# PipecatAssistant — Setup Reference

Reproducible setup for a fully-local voice assistant on Ubuntu 24.04 (Linux 6.17).
Stack: Pipecat + faster-whisper (STT) + Ollama `llama3.1:8b` (LLM) + Piper (TTS) + LocalAudioTransport (mic/speakers).

---

## 0. Hardware / OS baseline (reference install)
- Ubuntu 24.04, kernel 6.17
- NVIDIA RTX 4070 Laptop, 8 GB VRAM, driver 590.48.01
- Python 3.12.3 (system)

## 1. System packages (apt)

All system-level dependencies, in one command:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential python3-dev \
  portaudio19-dev libasound2-dev \
  espeak-ng
```

| Package | Why |
|---|---|
| `build-essential`, `python3-dev` | C compiler + Python headers — needed to build `pyaudio` wheel |
| `portaudio19-dev` | PortAudio headers for `pyaudio` (LocalAudioTransport mic/speaker) |
| `libasound2-dev` | ALSA headers (Linux audio backend) |
| `espeak-ng` | Phonemizer backend used by Piper voices |

## 2. NVIDIA / CUDA

Driver must be installed before you start (verify with `nvidia-smi`).
Pipecat's Whisper service uses `faster-whisper`, which brings its own CUDA runtime via pip wheels — no separate CUDA toolkit install required.

## 3. Python toolchain

```bash
# uv (Python package/project manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# re-open shell or: source ~/.local/bin/env
```

## 4. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
# Installs systemd unit; service auto-starts.
systemctl status ollama     # should show "active (running)"
curl http://localhost:11434 # should return "Ollama is running"
```

Pull the model we use:

```bash
ollama pull llama3.1:8b
```

## 5. Project bootstrap

```bash
cd /home/maikel/coding/PipecatAssistant
uv init --python 3.12 --no-workspace
uv add "pipecat-ai[whisper,ollama,piper,silero,local]" loguru python-dotenv websockets
uv add nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

The `pipecat-ai` extras:
- `whisper` — faster-whisper STT
- `ollama` — Ollama LLM client
- `piper` — Piper TTS client
- `silero` — Silero VAD (end-of-utterance detection)
- `local` — LocalAudioTransport (PyAudio mic/speakers)

Extra pip-level deps (not pulled transitively by Pipecat 1.0):
- `websockets` — imported at `stt_service` load time
- `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12` — CUDA runtime .so files for `faster-whisper` (ctranslate2). We preload them via `ctypes.CDLL(..., RTLD_GLOBAL)` in `bot.py` because `LD_LIBRARY_PATH` can't be changed in-process and the wheels drop their .so in `site-packages/nvidia/*/lib/`.

## 6. Piper voice model

```bash
mkdir -p models/piper
cd models/piper
# en_US, medium quality, female American voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ../..
```

### 6b. Other languages

The default persona is English. To run in Spanish, use the included `es` persona:

```bash
uv run python bot.py --persona es
```

It selects `whisper_model: SMALL` (multilingual — `DISTIL_MEDIUM_EN` is English-only) with `language: es`, plus the Piper voice `es_ES-davefx-medium`. On first launch the Whisper SMALL checkpoint and the Spanish voice auto-download into `models/` — expect ~15 s of one-time setup before the first reply.

To add another language: copy `wiki_templates/personas/es.md` to `<lang>.md`, change `language:`, `voice:`, `whisper_model:` (keep a multilingual model — `SMALL`, `MEDIUM`, `LARGE_V3_TURBO`), and rewrite the prompt. `llama3.1:8b` is natively multilingual, so no LLM change is needed.

## 7. Run

Only two moving parts — Pipecat 1.0's `PiperTTSService` loads the voice **in-process** via `PiperVoice.load()`, so no separate Piper HTTP server is needed.

```bash
# Term A — already running as systemd, but if manual:
ollama serve

# Term B — the bot
uv run python bot.py                          # defaults to local_audio
uv run python bot.py --transport local_audio  # same as default
uv run python bot.py --transport text         # text stdin/stdout, ~1s startup
uv run python bot.py --help                   # all CLI flags
```

Speak into the mic (local_audio) or type (text); you should get a reply.

CLI flags — all optional:

| flag | default | picks |
|---|---|---|
| `--transport` | `local_audio` | `text`, `local_audio`, `websocket` (stub) |
| `--user` | `maikel` | `wiki/people/<user>.md` |
| `--device` | depends on transport | `wiki/devices/<device>.md` |
| `--persona` | `default` | `wiki/personas/<persona>.md` (voice + system prompt) |

## 8. Wiki (agent memory)

First run auto-copies `wiki_templates/` → `wiki/` (gitignored). Edit the pages
to teach the assistant about yourself, your devices, and your personas.

Each turn:
- **Pre-LLM (retrieval)** — the active `personas/<persona>.md`, `devices/<device>.md`,
  `people/<user>.md`, plus today's **user statements** (extracted from the daily
  log) are injected into the LLM system prompt. Bot responses from prior turns
  are deliberately stripped to avoid a hallucination feedback loop.
- **Post-LLM (librarian)** — the turn is appended as a timestamped block to
  `wiki/daily/<today>.md` and a one-line summary to `wiki/log.md`.

Budget: `WIKI_INJECT_BUDGET_CHARS` (default 8000) caps the injected prompt.
Point `WIKI_DIR` at a different folder if you want to share a wiki across
checkouts or back it up to a synced drive.

Quick test that memory persists:
```bash
uv run python bot.py --transport text
> my morning routine is 10 situps, 5 jumps, 3 turns
> ^D                                    # Ctrl+D exits cleanly
uv run python bot.py --transport text   # fresh session
> what's my morning routine?
# → "10 situps, 5 jumps, 3 turns" (recalled from today's log)
```

---

## Verification snippets

```bash
# Core imports sanity
uv run python -c "import pipecat, pyaudio, faster_whisper, piper; print('ok')"

# Ollama is reachable
curl -s http://localhost:11434/api/tags | head

# Installed system libs
dpkg -l portaudio19-dev libasound2-dev espeak-ng build-essential python3-dev | grep ^ii
```

## Known gotchas
- **PyAudio build fails** → missing `build-essential` or `portaudio19-dev`. Install both.
- **Piper says "No module named espeakng"** → install `espeak-ng`.
- **VRAM pressure on 8 GB** → llama3.1:8b + Whisper may OOM. Fallbacks: `llama3.2:3b`, Whisper `base.en`.
- **Wrong mic/speaker picked** → set `input_device_index` / `output_device_index` on `LocalAudioTransport`. List devices: `uv run python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`.
- **`libcublas.so.12 not found`** (faster-whisper) → `uv add nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12`, then keep the `_preload_nvidia_libs()` ctypes shim in `bot.py`.
- **`ModuleNotFoundError: websockets`** at import time → `uv add websockets` (transitive dep missing in Pipecat 1.0.0).
- **VAD never fires (bot silent on speech)** → `vad_analyzer` must go on `LLMUserAggregatorParams`, NOT `LocalAudioTransportParams`. Pipecat 1.0 silently drops the kwarg on the transport.
- **Bot hears itself and interrupts** → add `AlwaysUserMuteStrategy()` to `LLMUserAggregatorParams.user_mute_strategies` so the mic is muted while the bot speaks.
