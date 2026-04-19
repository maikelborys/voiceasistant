"""PipecatAssistant entrypoint (thin shim).

Preloads CUDA libs, then delegates to voiceassistant.runner.main().

Kept callable as `uv run python bot.py` so existing SETUP.md / CLAUDE.md
instructions remain valid. CLI flags are forwarded:
    uv run python bot.py --transport local_audio   # v1 behaviour
    uv run python bot.py --transport text          # from step 5
    uv run python bot.py --help
"""

from voiceassistant.preload import preload_nvidia_libs

preload_nvidia_libs()

from voiceassistant.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
