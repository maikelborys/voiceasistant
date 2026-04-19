"""Semantic pre-LLM system-message injector.

Replaces the old structural `WikiRetrieval`. On `LLMContextFrame` (user
turn complete, LLM about to run) we embed the last user message, top-K
search the sqlite-vec index, and rebuild `messages[0]` as:

    <base persona prompt>
    --- pinned: personas/<id>.md ---
    --- pinned: devices/<id>.md ---
    --- pinned: people/<user>.md ---
    --- pinned: daily/YYYY-MM-DD.md (today's user statements) ---
    --- semantic: topics/coffee.md (score=0.62) ---
    --- semantic: notes/2026-04-15.md (score=0.57) ---

Pinned pages (identity + today's log) always go in regardless of score.
Semantic matches fill by descending score; any single match that would
bust the budget is skipped, but the loop continues so smaller
lower-ranked matches can still fit. If the embed call fails or the
index is empty, fall back to structural pages_for_session().

Daily logs (today's or any prior day that ranks semantically) are
always filtered to user-statements-only — bot responses never re-enter
the context, preventing a hallucination feedback loop.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voiceassistant.memory import embeddings, index
from voiceassistant.session import SessionContext
from voiceassistant.wiki.paths import wiki_dir
from voiceassistant.wiki.retriever import (
    _user_statements_from_daily,
    pages_for_session,
)
from voiceassistant.wiki.store import read_page

INDEX_FILENAME = "index.sqlite"


class VectorRetrieval(FrameProcessor):
    def __init__(
        self,
        session: SessionContext,
        context: LLMContext,
        base_system_prompt: str,
        budget_chars: int,
        k: int = 6,
    ) -> None:
        super().__init__()
        self._session = session
        self._context = context
        self._base_prompt = base_system_prompt
        self._budget = budget_chars
        self._k = k
        self._root: Path = wiki_dir()
        self._db: Path = self._root / INDEX_FILENAME
        self._pinned = (
            f"personas/{session.persona_id}.md",
            f"devices/{session.device_id}.md",
            f"people/{session.user_id}.md",
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, (LLMContextFrame, LLMRunFrame))
            and direction == FrameDirection.DOWNSTREAM
        ):
            await self._inject()
        await self.push_frame(frame, direction)

    def _last_user_text(self) -> str:
        for msg in reversed(self._context.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") in (None, "text")
                ]
                return " ".join(s for s in parts if s)
        return ""

    async def _semantic_hits(self, query: str) -> list[tuple[str, float]]:
        if not query or not self._db.exists():
            return []
        try:
            qvec = await embeddings.aembed(query)
        except Exception as e:
            logger.warning(f"VectorRetrieval: embed failed ({e}); falling back")
            return []
        try:
            with index.open_index(self._db) as conn:
                if index.count(conn) == 0:
                    return []
                return index.search(conn, qvec, k=self._k)
        except Exception as e:
            logger.warning(f"VectorRetrieval: search failed ({e}); falling back")
            return []

    def _load_body(self, relpath: str) -> str:
        """Read a page; daily logs get filtered to user-statements-only."""
        body = read_page(relpath)
        if not body or not body.strip():
            return ""
        if relpath.startswith("daily/"):
            body = _user_statements_from_daily(body)
        return body.strip()

    async def _inject(self) -> None:
        query = self._last_user_text()
        hits = await self._semantic_hits(query)

        # Pinned: identity pages + today's daily log (as user-only).
        today_iso = date.today().isoformat()
        today_path = f"daily/{today_iso}.md"
        pinned_relpaths = list(self._pinned) + [today_path]
        pinned_set = set(pinned_relpaths)

        loaded: list[tuple[str, str, float, str]] = []  # (label, body, score, kind)
        for relpath in pinned_relpaths:
            body = self._load_body(relpath)
            if body:
                loaded.append((relpath, body, 1.0, "pinned"))

        for relpath, score in hits:
            if relpath in pinned_set:
                continue
            body = self._load_body(relpath)
            if body:
                loaded.append((relpath, body, score, "semantic"))

        # Cold-start fallback: no pinned or semantic content at all.
        if not loaded:
            for label, body in pages_for_session(self._session):
                loaded.append((label, body, 0.0, "fallback"))

        self._write_messages(loaded, len(hits))

    def _write_messages(
        self,
        loaded: list[tuple[str, str, float, str]],
        n_hits: int,
    ) -> None:
        remaining = self._budget - len(self._base_prompt)

        # Pinned first, then semantic by descending score.
        pinned = [x for x in loaded if x[3] == "pinned"]
        rest = sorted(
            (x for x in loaded if x[3] != "pinned"), key=lambda x: -x[2]
        )

        blocks: list[str] = []
        for label, body, score, kind in pinned + rest:
            tag = (
                f"pinned: {label}"
                if kind == "pinned"
                else (
                    f"semantic: {label} (score={score:.2f})"
                    if kind == "semantic"
                    else f"fallback: {label}"
                )
            )
            block = f"\n\n--- {tag} ---\n{body}"
            if len(block) > remaining:
                if kind == "pinned":
                    # Pinned must be present — truncate and stop adding more.
                    blocks.append(block[: max(remaining, 0)] + "\n...[truncated]")
                    remaining = 0
                    continue
                # Semantic match too big; skip and try the next (smaller) one.
                continue
            blocks.append(block)
            remaining -= len(block)

        if blocks:
            header = (
                "\n\nThe pages below are your memory for this user and session. "
                "Treat them as factual when answering questions about the user, "
                "their preferences, or prior conversations. When they are silent, "
                "answer from your own knowledge while staying in character."
            )
            new_system = self._base_prompt + header + "".join(blocks)
        else:
            new_system = self._base_prompt
        messages = self._context.messages
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": new_system}
        else:
            messages.insert(0, {"role": "system", "content": new_system})
        logger.debug(
            f"VectorRetrieval: injected {len(new_system)} chars "
            f"({len(blocks)} blocks, {n_hits} semantic hits, "
            f"{sum(1 for _,_,_,k in loaded if k=='semantic')} semantic eligible)"
        )
