"""Build / refresh the sqlite-vec index over the wiki.

Two entry points:

- `ensure_fresh(wiki_root)` — called on every startup from `runner.py`.
  Cheap when nothing changed (one stat per file). Re-embeds only the
  .md files whose mtime moved past the value stored in the index, plus
  drops rows for files that no longer exist.

- `python -m voiceassistant.memory.build_index [--rebuild]` — CLI for
  manual reindex / full wipe.

Skipped paths: anything inside `.history/`, the index files themselves,
and non-markdown files. Whole-file embedding (no chunking) — nomic-embed
has 8192-token context which covers any reasonable Markdown page; daily
logs are cheap to re-embed each time they grow.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from voiceassistant.memory import embeddings, index
from voiceassistant.wiki.paths import wiki_dir
from voiceassistant.wiki.retriever import _user_statements_from_daily

INDEX_FILENAME = "index.sqlite"
_SKIP_DIR_NAMES = {".history", ".obsidian"}
# log.md is the append-only turn-audit trail (contains bot responses) — never
# index it; would reintroduce hallucinated bot claims into retrieval context.
_SKIP_FILES = {"log.md"}
# nomic-embed-text in Ollama rejects inputs past its context window with a
# 400 before respecting num_ctx overrides. Empirically ~5.8k chars trips it;
# cap well under that with headroom for multi-byte scripts.
_MAX_EMBED_CHARS = 3500


def _cap_for_embed(relpath: str, text: str) -> str:
    if len(text) <= _MAX_EMBED_CHARS:
        return text
    # Daily logs grow chronologically — keep the tail so recent statements win.
    if relpath.startswith("daily/"):
        return text[-_MAX_EMBED_CHARS:]
    return text[:_MAX_EMBED_CHARS]


def _index_path(root: Path) -> Path:
    return root / INDEX_FILENAME


def _iter_markdown(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.md"):
        rel = p.relative_to(root)
        if any(part in _SKIP_DIR_NAMES for part in rel.parts):
            continue
        if str(rel) in _SKIP_FILES:
            continue
        out.append(p)
    return out


async def _rebuild(root: Path, full: bool = False) -> tuple[int, int]:
    db = _index_path(root)
    if full and db.exists():
        db.unlink()
    files = _iter_markdown(root)
    relpaths = [str(p.relative_to(root)) for p in files]

    with index.open_index(db) as conn:
        existing_paths = {
            row[0]
            for row in conn.execute("SELECT path FROM pages").fetchall()
        }
        # Drop rows for files that no longer exist.
        current = set(relpaths)
        for stale in existing_paths - current:
            index.delete(conn, stale)

        to_embed: list[tuple[str, Path, float, str]] = []
        for relpath, path in zip(relpaths, files):
            mtime = path.stat().st_mtime
            if not index.page_needs_reindex(conn, relpath, mtime):
                continue
            text = path.read_text(encoding="utf-8")
            # Daily logs: index only user statements, never bot responses
            # (avoids hallucination feedback loop — same rule as retriever.py).
            if relpath.startswith("daily/"):
                text = _user_statements_from_daily(text)
            if not text.strip():
                continue
            text = _cap_for_embed(relpath, text)
            to_embed.append((relpath, path, mtime, text))

        if not to_embed:
            return 0, len(existing_paths)

        vecs = await embeddings.aembed_many([t[3] for t in to_embed])
        for (relpath, path, mtime, text), vec in zip(to_embed, vecs):
            index.upsert(conn, relpath, vec, mtime=mtime, chars=len(text))
        return len(to_embed), index.count(conn)


async def ensure_fresh(root: Path | None = None) -> None:
    """Called from runner.py on startup. Idempotent; logs churn only."""
    root = root or wiki_dir()
    if not root.exists():
        return
    embedded, total = await _rebuild(root, full=False)
    if embedded:
        logger.info(f"wiki index: (re)embedded {embedded} pages, {total} total")
    else:
        logger.debug(f"wiki index: {total} pages, no changes")


async def upsert_path(root: Path, relpath: str) -> None:
    """Called by the librarian after it writes a single page."""
    full = root / relpath
    if not full.exists():
        with index.open_index(_index_path(root)) as conn:
            index.delete(conn, relpath)
        return
    text = full.read_text(encoding="utf-8")
    if relpath.startswith("daily/"):
        text = _user_statements_from_daily(text)
    if not text.strip():
        return
    text = _cap_for_embed(relpath, text)
    vec = await embeddings.aembed(text)
    with index.open_index(_index_path(root)) as conn:
        index.upsert(
            conn, relpath, vec, mtime=full.stat().st_mtime, chars=len(text)
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description="(Re)build the wiki vector index.")
    parser.add_argument(
        "--rebuild", action="store_true", help="wipe index first, then rebuild"
    )
    parser.add_argument(
        "--wiki", type=Path, default=None, help="override wiki dir"
    )
    args = parser.parse_args()
    root = args.wiki or wiki_dir()
    embedded, total = asyncio.run(_rebuild(root, full=args.rebuild))
    print(f"embedded={embedded} total={total} db={_index_path(root)}")


if __name__ == "__main__":
    _main()
