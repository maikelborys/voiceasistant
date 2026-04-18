# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fully-local, on-device voice assistant. Mic → Silero VAD → faster-whisper (CUDA) → Ollama `llama3.1:8b` → Piper TTS (CPU) → speakers. Orchestrated by Pipecat 1.0. No cloud calls — privacy constraint is a hard requirement, do not propose OpenAI/Anthropic/Google STT/LLM/TTS.

Full install story (apt packages, uv commands, model downloads, known gotchas) lives in `SETUP.md`. Keep it updated when system-level deps change.

## Commands

```bash
# Run the assistant (requires ollama systemd unit active)
uv run python bot.py

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

There are no tests and no lint config — this is a single-file prototype. Don't invent a test suite unless explicitly asked.

## Architecture

Single entrypoint `bot.py`. Everything wires up inside `main()`:

1. **`_preload_nvidia_libs()` runs before any pipecat import.** faster-whisper's `ctranslate2` backend `dlopen`s `libcublas.so.12` / `libcudnn*.so` from the `nvidia-*-cu12` pip wheels. Those `.so` files are in `site-packages/nvidia/*/lib/` which is NOT on `LD_LIBRARY_PATH`, and you cannot mutate `LD_LIBRARY_PATH` in-process. The fix is `ctypes.CDLL(..., RTLD_GLOBAL)` preloading in dependency order (cuda_runtime → cublas → cuda_nvrtc → cudnn). Do not move this call below the imports — it has to win the race against ctranslate2's own dlopen.

2. **`SpeechEventLogger`** is a `FrameProcessor` placed right after `transport.input()`. It logs VAD / user-speaking / bot-speaking / mute-state / transcription frames with direction arrows (↑/↓). It's the fastest way to tell whether the pipeline is stuck. Keep it even when everything works — debugging without it is guesswork.

3. **Pipeline order is load-bearing**:
   ```
   transport.input() → SpeechEventLogger → stt → aggregators.user() → llm → tts → transport.output() → aggregators.assistant()
   ```
   `aggregators.user()` must be between STT and LLM. `aggregators.assistant()` sits after `transport.output()` so it sees bot-speaking frames in the right order.

4. **Pipecat 1.0 API quirks that silently bite**:
   - Use `LLMContext` + `LLMContextAggregatorPair` from `pipecat.processors.aggregators.llm_response_universal`. The older `OpenAILLMContext` module is gone.
   - `vad_analyzer` goes on `LLMUserAggregatorParams`, NOT `LocalAudioTransportParams`. Pipecat 1.0 silently drops it if you pass it to the transport — the symptom is audio captured but VAD never fires.
   - `user_mute_strategies=[AlwaysUserMuteStrategy()]` on `LLMUserAggregatorParams` is required to stop the bot hearing itself through the speakers. Without it the bot interrupts its own speech.
   - Services use kwargs-only `Settings` dataclass: `OLLamaLLMService(settings=OLLamaLLMService.Settings(model="...", temperature=0.7))`.

5. **Piper is in-process.** `PiperTTSService` loads the voice via `PiperVoice.load()` — there is no separate `piper.http_server` process. `download_dir` must point at a directory containing `<voice>.onnx` + `.onnx.json`; auto-downloads if missing. `use_cuda=False` is intentional (we don't install `onnxruntime-gpu`; CPU is fast enough for Piper).

## Memory and plan

Cross-session context lives outside the repo:
- `~/.claude/projects/-home-maikel-coding-PipecatAssistant/memory/` — auto-loaded per project, has gotchas + v1 status
- `~/.claude/plans/sequential-swimming-lake.md` — phase-based tracker with `[x]/[ ]/[~]` markers. v1 is green as of 2026-04-17. Next-session backlog is Phase 5.2 (polish) and Phase 6 (tool-calling, persistent context, wake-word, web UI).

Before implementation work, read the plan and confirm alignment with the user — he prefers plan-first, then step-by-step execution.

## Hardware constraint

RTX 4070 Laptop, 8 GB VRAM. llama3.1:8b + Whisper DISTIL_MEDIUM_EN co-resident is tight. If OOM, fall back to `llama3.2:3b` or Whisper `BASE_EN` rather than swapping to cloud services.
