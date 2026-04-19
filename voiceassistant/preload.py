"""Preload CUDA shared libraries before any pipecat / ctranslate2 import.

ctranslate2 (under faster-whisper) dlopen's cuBLAS / cuDNN via the loader. The
`nvidia-*-cu12` pip wheels drop their .so files in site-packages/nvidia/*/lib/,
which is not on LD_LIBRARY_PATH. RTLD_GLOBAL preloading makes them visible to
later dlopen calls. Order matters: dependencies before dependents.

Must be called *before* anything imports pipecat/ctranslate2/faster-whisper.
"""

from __future__ import annotations

import ctypes
import site
from pathlib import Path

_PRIORITY_ORDER = ("cuda_runtime", "cublas", "cuda_nvrtc", "cudnn")


def preload_nvidia_libs() -> None:
    search_dirs: list[Path] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pkg in _PRIORITY_ORDER:
            search_dirs.extend(Path(sp).glob(f"nvidia/{pkg}/lib"))

    for d in search_dirs:
        for lib in sorted(d.glob("*.so*")):
            if ".alt." in lib.name:
                continue
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
