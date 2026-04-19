"""web_search tool — DuckDuckGo text search via the `ddgs` library.

No API key required. We cap results and total output length because the
response goes back into the LLM context and counts against token budget.
"""

from __future__ import annotations

import asyncio

from ddgs import DDGS
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

NAME = "web_search"
MAX_RESULTS = 5
MAX_CHARS = 900

SCHEMA = FunctionSchema(
    name=NAME,
    description=(
        "Search the public web via DuckDuckGo for up-to-date information. "
        "Use this for current events, weather, news, or any fact that is "
        "likely to have changed recently. Returns a short list of titles, "
        "URLs, and snippets."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "Plain-language search query, e.g. 'weather Barcelona today'.",
        },
    },
    required=["query"],
)


def _run_search(query: str) -> list[dict]:
    return list(DDGS().text(query, max_results=MAX_RESULTS))


def _format(results: list[dict]) -> str:
    if not results:
        return "(no results)"
    lines: list[str] = []
    for r in results:
        title = (r.get("title") or "").strip()
        href = (r.get("href") or "").strip()
        body = (r.get("body") or "").replace("\n", " ").strip()
        entry = f"- {title} — {href}\n  {body}"
        lines.append(entry)
    out = "\n".join(lines)
    if len(out) > MAX_CHARS:
        out = out[: MAX_CHARS - 15] + "…[truncated]"
    return out


async def handler(params: FunctionCallParams) -> None:
    query = (params.arguments or {}).get("query", "").strip()
    if not query:
        await params.result_callback({"result": "empty query"})
        return
    logger.info(f"web_search: '{query}'")
    try:
        results = await asyncio.to_thread(_run_search, query)
    except Exception as e:
        logger.warning(f"web_search failed: {e}")
        await params.result_callback({"result": f"search failed: {e}"})
        return
    await params.result_callback({"result": _format(results)})
