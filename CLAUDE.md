# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fully-local, on-device voice assistant. Mic → Silero VAD → faster-whisper (CUDA) → Ollama `llama3.1:8b` → Piper TTS (CPU) → speakers. Orchestrated by Pipecat 1.0. **Local by default; cloud LLM is opt-in per run via `--llm openrouter/<model>`** (requires `OPENROUTER_API_KEY`). STT and TTS must stay local (audio privacy); do not propose OpenAI/Anthropic/Google STT/TTS.

Full install story (apt packages, uv commands, model downloads, known gotchas) lives in `SETUP.md`. Keep it updated when system-level deps change.

## Commands

```bash
# Run the assistant (requires ollama systemd unit active)
uv run python bot.py                              # defaults to local_audio
uv run python bot.py --transport text             # text stdin/stdout — fast debug
uv run python bot.py --help                       # all flags
OPENROUTER_API_KEY=sk-or-... uv run python bot.py --transport text \
    --llm openrouter/anthropic/claude-sonnet-4.5  # cloud LLM, local STT/TTS still

# Microphone diagnostic — prints RMS per input device so we know the mic is live
uv run python mic_probe.py
uv run python mic_probe.py --index 7   # probe a single device

# Dependency management
uv add <pkg>            # adds to pyproject.toml + uv.lock
uv sync                 # reinstall from lock

# Ollama sanity
systemctl status ollama
curl -s http://localhost:11434/api/tags
```

There are no tests and no lint config. Don't invent a test suite unless explicitly asked.

## Architecture

`bot.py` is a 4-line shim: preload NVIDIA libs → `voiceassistant.runner.main()`.
All real code lives in the `voiceassistant/` package:

```
voiceassistant/
├── preload.py               # ctypes CDLL(RTLD_GLOBAL) for CUDA libs — MUST run before any pipecat import
├── session.py               # SessionContext (session_id, device_id, user_id, persona_id, transport_kind)
├── config.py                # env-overridable defaults
├── runner.py                # CLI parse → session → bundle → pipeline → run
├── pipeline.py              # build_pipeline(session, bundle, persona) — transport-agnostic
├── personas.py              # load_persona() — parses wiki/personas/<id>.md
├── audio_devices.py         # shared PyAudio enumeration (also used by mic_probe.py)
├── transports/
│   ├── __init__.py          # TransportBundle + make_transport(session)
│   ├── local_audio.py       # pipecat LocalAudioTransport bundle
│   ├── text.py              # StdinTextInput + StdoutTextOutput bundle
│   └── websocket.py         # stub (Phase 9, ESP32)
├── processors/
│   ├── speech_logger.py     # direction-arrow frame logger — KEEP IN PIPELINE
│   ├── wiki_retrieval.py    # pre-LLM system-prompt injector
│   └── wiki_librarian.py    # pipeline-tail daily-log appender
└── wiki/
    ├── paths.py             # wiki_dir() + ensure_wiki_seeded() (copies wiki_templates/)
    ├── store.py             # read_page / write_page / append_page
    ├── retriever.py         # pages_for_session() — persona + device + user + today's user statements
    └── librarian.py         # append_daily_log() — structured turn blocks + log.md line
```

Key load-bearing details:

1. **`preload_nvidia_libs()` runs before any pipecat import.** faster-whisper's `ctranslate2` backend `dlopen`s `libcublas.so.12` / `libcudnn*.so` from the `nvidia-*-cu12` pip wheels. Those `.so` files are in `site-packages/nvidia/*/lib/` which is NOT on `LD_LIBRARY_PATH`, and you cannot mutate `LD_LIBRARY_PATH` in-process. The fix is `ctypes.CDLL(..., RTLD_GLOBAL)` preloading in dependency order (cuda_runtime → cublas → cuda_nvrtc → cudnn). `bot.py` calls it first thing; do not move it below the `voiceassistant.runner` import.

2. **`SpeechEventLogger`** sits right after `bundle.input` and logs VAD / speech-lifecycle / mute / transcription frames with direction arrows (↑/↓). Fastest way to tell whether the pipeline is stuck.

3. **Pipeline order is load-bearing** (built in `voiceassistant/pipeline.py`):
   ```
   bundle.input → SpeechEventLogger → [stt] → aggregators.user() → WikiRetrieval
     → llm → [tts] → bundle.output → aggregators.assistant() → WikiLibrarian
   ```
   STT/TTS/VAD/user-mute are gated on `TransportBundle` flags — local_audio enables all four, text disables all four. `aggregators.user()` must sit between STT and LLM. `WikiRetrieval` mutates `messages[0]` on every `UserStartedSpeakingFrame`. `WikiLibrarian` writes on `LLMContextAssistantTimestampFrame` (the downstream-facing commit signal — `LLMFullResponseEndFrame` is consumed upstream by the assistant aggregator).

4. **Pipecat 1.0 API quirks that silently bite**:
   - Use `LLMContext` + `LLMContextAggregatorPair` from `pipecat.processors.aggregators.llm_response_universal`. The older `OpenAILLMContext` module is gone.
   - `vad_analyzer` goes on `LLMUserAggregatorParams`, NOT `LocalAudioTransportParams`. Pipecat 1.0 silently drops it if you pass it to the transport — the symptom is audio captured but VAD never fires.
   - `user_mute_strategies=[AlwaysUserMuteStrategy()]` on `LLMUserAggregatorParams` is required to stop the bot hearing itself through the speakers. Without it the bot interrupts its own speech.
   - Services use kwargs-only `Settings` dataclass: `OLLamaLLMService(settings=OLLamaLLMService.Settings(model="...", temperature=0.7))`.

5. **Piper is in-process.** `PiperTTSService` loads the voice via `PiperVoice.load()` — there is no separate `piper.http_server` process. `download_dir` must point at a directory containing `<voice>.onnx` + `.onnx.json`; auto-downloads if missing. `use_cuda=False` is intentional (we don't install `onnxruntime-gpu`; CPU is fast enough for Piper).

6. **Wiki auto-seeds on first run.** `ensure_wiki_seeded()` copies `wiki_templates/` (tracked) → `wiki/` (gitignored). Edit pages freely; never commit `wiki/`. Override location with `WIKI_DIR=/path/to/wiki`.

7. **Retrieval strips bot lines from daily log.** `wiki/retriever.py` injects persona/device/user pages + *today's user statements only* (as a bulleted list with timestamps). We deliberately do NOT feed the bot's prior responses back into the system prompt — that would be a hallucination feedback loop (one wrong answer becomes "truth" for the next session). User statements are the closest thing to durable facts until an LLM librarian promotes them to `people/<user>.md` (Phase 8.3). `WIKI_INJECT_BUDGET_CHARS` defaults to 8000.

8. **Text transport uses `asyncio.StreamReader` + `EndTaskFrame` upstream.** `run_in_executor(sys.stdin.readline)` is uncancellable (blocked thread on `read()`) and leaves dangling tasks on Ctrl+C. `EndFrame` pushed downstream didn't actually stop the pipeline — `EndTaskFrame` pushed UPSTREAM is the graceful shutdown signal.

9. **Identity lives at the transport layer, not in the LLM.** Every session carries `device_id` / `user_id` / `persona_id`; the LLM never decides who it's talking to. For multi-toy (Phase 9), each WebSocket connection will carry those fields in query params.

## Memory and plan

Cross-session context lives outside the repo:
- `~/.claude/projects/-home-maikel-coding-PipecatAssistant/memory/` — auto-loaded per project, has gotchas + v1 status
- `~/.claude/plans/jolly-nibbling-octopus.md` — Phase 8 MVP plan (runner + text transport + wiki retrieval + librarian). Completed 2026-04-19.

Before implementation work, read the plan and confirm alignment with the user — he prefers plan-first, then step-by-step execution.

## Hardware constraint

RTX 4070 Laptop, 8 GB VRAM. llama3.1:8b + Whisper DISTIL_MEDIUM_EN co-resident is tight. If OOM, fall back to `llama3.2:3b` or Whisper `BASE_EN` rather than swapping to cloud services.
