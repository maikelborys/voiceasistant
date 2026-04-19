"""Wiki + templates path resolution.

Default wiki location is `<project>/wiki/` (gitignored). Set `WIKI_DIR` to
point elsewhere — useful when sharing wikis across checkouts or backing up
to a synced folder.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from voiceassistant import config


def wiki_dir() -> Path:
    override = os.environ.get("WIKI_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return config.PROJECT_ROOT / "wiki"


def templates_dir() -> Path:
    return config.PROJECT_ROOT / "wiki_templates"


def ensure_wiki_seeded() -> Path:
    """Copy wiki_templates/ → wiki/ on first run. Idempotent; never overwrites."""
    target = wiki_dir()
    if target.exists():
        return target
    src = templates_dir()
    if not src.exists():
        raise FileNotFoundError(
            f"wiki_templates/ missing at {src} — reinstall or git checkout lost it"
        )
    shutil.copytree(src, target)
    return target
