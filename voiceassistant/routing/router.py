"""ModelRouter — per-turn LLM backend/model override driven by wiki/routing.md.

Sits between `aggregators.user()` and the LLM. On every `LLMContextFrame` /
`LLMRunFrame` going downstream, it:

  1. Reads the latest user message from the shared `LLMContext`.
  2. Evaluates `wiki/routing.md` rules (re-parsed when the file's mtime
     changes). First regex match wins.
  3. If the matched spec differs from the LLM service's current spec,
     reconfigures the live service in place (model name + OpenAI client
     base_url/api_key), then passes the frame through unchanged.
  4. When no rule matches, reverts to the default spec supplied at pipeline
     build time.

Cross-backend swaps work because OLLamaLLMService and OpenAILLMService both
extend BaseOpenAILLMService, holding a mutable `_client` (AsyncOpenAI) and
`_settings.model` — we swap both atomically.

Keyword matching is case-insensitive `re.search`; patterns live per-line in
`wiki/routing.md` as `<regex> -> <spec>`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.frames.frames import Frame, LLMContextFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService

from voiceassistant import config
from voiceassistant.llm_factory import ResolvedLLM, parse_spec
from voiceassistant.wiki.paths import wiki_dir

ROUTING_FILENAME = "routing.md"


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    spec: str


class ModelRouter(FrameProcessor):
    def __init__(self, llm: LLMService, default_spec: str) -> None:
        super().__init__()
        self._llm = llm
        self._default_spec = default_spec
        self._current_spec = default_spec
        self._rules_path: Path = wiki_dir() / ROUTING_FILENAME
        self._rules: list[Rule] = []
        self._rules_mtime: float = -1.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, (LLMContextFrame, LLMRunFrame))
            and direction == FrameDirection.DOWNSTREAM
        ):
            try:
                self._maybe_reroute(frame)
            except Exception as e:
                # Never let routing kill the turn — fall back to whatever's
                # already live on the LLM service.
                logger.warning(f"ModelRouter: reroute failed ({e}); keeping current spec")
        await self.push_frame(frame, direction)

    # --- rule loading ---

    def _load_rules(self) -> None:
        if not self._rules_path.exists():
            self._rules = []
            self._rules_mtime = -1.0
            return
        mtime = self._rules_path.stat().st_mtime
        if mtime == self._rules_mtime:
            return
        self._rules_mtime = mtime
        rules: list[Rule] = []
        in_fence = False
        for raw in self._rules_path.read_text().splitlines():
            line = raw.strip()
            # Skip everything inside ```...``` fenced blocks — routing.md uses
            # them for format/example docs that would otherwise parse as rules.
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not line or line.startswith("#"):
                continue
            if "->" not in line:
                continue
            # Prose lines in routing.md often mention `->` wrapped in
            # backticks ("Whitespace around `->` is ignored."). Real rules
            # are plain text, so a line containing backticks is doc, not rule.
            if "`" in line:
                continue
            pattern_str, _, spec_str = line.partition("->")
            pattern_str = pattern_str.strip()
            spec_str = spec_str.strip()
            if not pattern_str or not spec_str:
                continue
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"ModelRouter: bad regex {pattern_str!r}: {e}")
                continue
            rules.append(Rule(pattern=pattern, spec=spec_str))
        self._rules = rules
        if rules:
            logger.info(f"ModelRouter: loaded {len(rules)} routing rule(s)")

    # --- matching + reroute ---

    def _maybe_reroute(self, frame: Frame) -> None:
        self._load_rules()
        query = self._last_user_text(frame)
        target_spec = self._default_spec
        if query and self._rules:
            for rule in self._rules:
                if rule.pattern.search(query):
                    target_spec = rule.spec
                    logger.info(
                        f"ModelRouter: '{rule.pattern.pattern}' matched → {target_spec}"
                    )
                    break
        if target_spec == self._current_spec:
            return
        try:
            resolved = parse_spec(target_spec)
        except ValueError as e:
            logger.warning(f"ModelRouter: {e}; staying on {self._current_spec}")
            return
        if resolved.backend == "openrouter" and not config.OPENROUTER_API_KEY:
            logger.warning(
                "ModelRouter: OPENROUTER_API_KEY unset; skipping reroute "
                f"to {target_spec}"
            )
            return
        self._reconfigure(resolved)
        self._current_spec = target_spec

    def _last_user_text(self, frame: Frame) -> str:
        ctx: LLMContext | None = getattr(frame, "context", None)
        if ctx is None:
            return ""
        for msg in reversed(ctx.messages):
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

    def _reconfigure(self, resolved: ResolvedLLM) -> None:
        """Swap the live LLM service's model + OpenAI client in place."""
        if resolved.backend == "ollama":
            api_key = "ollama"
            base_url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/v1"
        elif resolved.backend == "openrouter":
            api_key = config.OPENROUTER_API_KEY
            base_url = config.OPENROUTER_BASE_URL
        else:
            return
        self._llm._settings.model = resolved.model
        self._llm._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=1000,
                    keepalive_expiry=None,
                )
            ),
        )
        logger.info(
            f"ModelRouter: reconfigured llm → backend={resolved.backend} "
            f"model={resolved.model}"
        )
