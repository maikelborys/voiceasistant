"""Ollama embeddings client — nomic-embed-text → 768-d float vectors.

Uses the `/api/embed` endpoint (batched; supersedes `/api/embeddings`).
Both sync and async entry points; retrieval runs on the pipeline's event
loop so `aembed` is the hot path, build_index uses the sync wrapper.
"""

from __future__ import annotations

import asyncio

import httpx

from voiceassistant import config

EMBED_DIM = 768  # nomic-embed-text dimension


async def aembed(text: str, *, client: httpx.AsyncClient | None = None) -> list[float]:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    finally:
        if own_client:
            await client.aclose()


async def aembed_many(texts: list[str]) -> list[list[float]]:
    """Batch embed — one HTTP call, Ollama handles the batch."""
    if not texts:
        return []
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def embed(text: str) -> list[float]:
    return asyncio.run(aembed(text))


def embed_many(texts: list[str]) -> list[list[float]]:
    return asyncio.run(aembed_many(texts))
