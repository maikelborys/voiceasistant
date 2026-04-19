"""sqlite-vec wrapper — top-K semantic search over wiki pages.

Layout:

    pages      (id INTEGER PK, path TEXT UNIQUE, mtime REAL, chars INTEGER)
    pages_vec  virtual vec0 (embedding FLOAT[768])  — rowid matches pages.id

Kept deliberately small. Embedding is caller-supplied — the index is
unaware of whichever model produced it. The search API returns normalised
similarity scores in [0, 1] (higher = closer).
"""

from __future__ import annotations

import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from voiceassistant.memory.embeddings import EMBED_DIM


def _floats_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


@contextmanager
def open_index(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _ensure_schema(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " path TEXT UNIQUE NOT NULL,"
        " mtime REAL NOT NULL,"
        " chars INTEGER NOT NULL"
        ")"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS pages_vec USING vec0("
        f" embedding FLOAT[{EMBED_DIM}]"
        f")"
    )


def page_needs_reindex(conn: sqlite3.Connection, path: str, mtime: float) -> bool:
    row = conn.execute("SELECT mtime FROM pages WHERE path = ?", (path,)).fetchone()
    return row is None or row[0] < mtime


def upsert(
    conn: sqlite3.Connection,
    path: str,
    embedding: list[float],
    mtime: float,
    chars: int,
) -> None:
    if len(embedding) != EMBED_DIM:
        raise ValueError(f"embedding dim {len(embedding)} != {EMBED_DIM}")
    blob = _floats_to_blob(embedding)
    cur = conn.execute("SELECT id FROM pages WHERE path = ?", (path,))
    row = cur.fetchone()
    if row is None:
        cur = conn.execute(
            "INSERT INTO pages (path, mtime, chars) VALUES (?, ?, ?)",
            (path, mtime, chars),
        )
        page_id = cur.lastrowid
        conn.execute(
            "INSERT INTO pages_vec (rowid, embedding) VALUES (?, ?)",
            (page_id, blob),
        )
    else:
        page_id = row[0]
        conn.execute(
            "UPDATE pages SET mtime = ?, chars = ? WHERE id = ?",
            (mtime, chars, page_id),
        )
        conn.execute("DELETE FROM pages_vec WHERE rowid = ?", (page_id,))
        conn.execute(
            "INSERT INTO pages_vec (rowid, embedding) VALUES (?, ?)",
            (page_id, blob),
        )


def delete(conn: sqlite3.Connection, path: str) -> None:
    row = conn.execute("SELECT id FROM pages WHERE path = ?", (path,)).fetchone()
    if row is None:
        return
    page_id = row[0]
    conn.execute("DELETE FROM pages_vec WHERE rowid = ?", (page_id,))
    conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))


def search(
    conn: sqlite3.Connection, query_embedding: list[float], k: int = 6
) -> list[tuple[str, float]]:
    """Return [(path, similarity)] — similarity in [0, 1], higher = closer."""
    if len(query_embedding) != EMBED_DIM:
        raise ValueError(f"query dim {len(query_embedding)} != {EMBED_DIM}")
    blob = _floats_to_blob(query_embedding)
    rows = conn.execute(
        "SELECT pages.path, pages_vec.distance"
        " FROM pages_vec"
        " JOIN pages ON pages.id = pages_vec.rowid"
        " WHERE pages_vec.embedding MATCH ? AND k = ?"
        " ORDER BY pages_vec.distance",
        (blob, k),
    ).fetchall()
    # sqlite-vec returns L2 distance; convert to a bounded similarity for logging.
    return [(path, 1.0 / (1.0 + dist)) for path, dist in rows]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
