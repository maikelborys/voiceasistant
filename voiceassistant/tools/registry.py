"""Tool registry — maps tool names to (schema, handler) and filters by persona allowlist.

`build_tools(allowed)` returns a ToolsSchema for the LLMContext and a
`{name: handler}` dict the pipeline registers with the LLM service.
Unknown names in the allowlist are logged and skipped.
"""

from __future__ import annotations

from typing import Callable, Iterable

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from voiceassistant.tools import memory_search, save_note, web_search

_ALL: dict[str, tuple[FunctionSchema, Callable]] = {
    web_search.NAME: (web_search.SCHEMA, web_search.handler),
    memory_search.NAME: (memory_search.SCHEMA, memory_search.handler),
    save_note.NAME: (save_note.SCHEMA, save_note.handler),
}


def available_tool_names() -> tuple[str, ...]:
    return tuple(_ALL.keys())


def build_tools(
    allowed: Iterable[str],
) -> tuple[ToolsSchema | None, dict[str, Callable]]:
    """Return (ToolsSchema, handler_map). Both None/empty if allowed is empty."""
    schemas: list[FunctionSchema] = []
    handlers: dict[str, Callable] = {}
    for name in allowed:
        entry = _ALL.get(name)
        if entry is None:
            logger.warning(f"tool registry: unknown tool '{name}' in allowlist — skipping")
            continue
        schema, handler = entry
        schemas.append(schema)
        handlers[name] = handler
    if not schemas:
        return None, {}
    return ToolsSchema(standard_tools=schemas), handlers
