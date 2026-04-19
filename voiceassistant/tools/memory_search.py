"""memory_search tool — semantic search over the wiki vector index.

Complements the per-turn pinned/semantic retrieval: the LLM can
explicitly pull more context when pinned pages are silent on the topic.
"""

from __future__ import annotations

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from voiceassistant.memory import embeddings, index
from voiceassistant.memory.build_index import INDEX_FILENAME
from voiceassistant.wiki.paths import wiki_dir
from voiceassistant.wiki.retriever import _user_statements_from_daily
from voiceassistant.wiki.store import read_page

NAME = "memory_search"
DEFAULT_K = 3
EXCERPT_CHARS = 400
MAX_CHARS = 1500

SCHEMA = FunctionSchema(
    name=NAME,
    description=(
        "Search the user's private wiki (past conversations, notes, topic "
        "pages) for anything relevant to a query. Use this when the user "
        "refers to something they said earlier and the answer isn't already "
        "visible to you. Returns short excerpts with their source page."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "What to look up, in plain language.",
        },
        "k": {
            "type": "integer",
            "description": "Max number of pages to return (default 3, cap 6).",
        },
    },
    required=["query"],
)


def _excerpt(body: str) -> str:
    body = body.strip()
    if len(body) <= EXCERPT_CHARS:
        return body
    return body[: EXCERPT_CHARS - 1] + "…"


async def handler(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    query = str(args.get("query", "")).strip()
    k = int(args.get("k") or DEFAULT_K)
    k = max(1, min(k, 6))
    if not query:
        await params.result_callback({"result": "empty query"})
        return

    db = wiki_dir() / INDEX_FILENAME
    if not db.exists():
        await params.result_callback({"result": "(index not built)"})
        return

    try:
        qvec = await embeddings.aembed(query)
    except Exception as e:
        logger.warning(f"memory_search embed failed: {e}")
        await params.result_callback({"result": f"embed failed: {e}"})
        return

    try:
        with index.open_index(db) as conn:
            hits = index.search(conn, qvec, k=k)
    except Exception as e:
        logger.warning(f"memory_search lookup failed: {e}")
        await params.result_callback({"result": f"search failed: {e}"})
        return

    if not hits:
        await params.result_callback({"result": "(no matches)"})
        return

    blocks: list[str] = []
    total = 0
    for relpath, score in hits:
        body = read_page(relpath) or ""
        if relpath.startswith("daily/"):
            body = _user_statements_from_daily(body)
        body = _excerpt(body)
        if not body:
            continue
        block = f"- {relpath} (score={score:.2f})\n{body}"
        if total + len(block) > MAX_CHARS:
            break
        blocks.append(block)
        total += len(block)

    out = "\n\n".join(blocks) or "(no matches)"
    logger.info(f"memory_search: '{query}' → {len(blocks)} hit(s)")
    await params.result_callback({"result": out})
