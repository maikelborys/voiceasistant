"""Minimal wiki file I/O — read, write, append."""

from __future__ import annotations

from voiceassistant.wiki.paths import wiki_dir


def read_page(relative: str) -> str | None:
    p = wiki_dir() / relative
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def write_page(relative: str, content: str) -> None:
    p = wiki_dir() / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def append_page(relative: str, content: str) -> None:
    p = wiki_dir() / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
