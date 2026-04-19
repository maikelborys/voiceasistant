"""End-of-session LLM librarian.

Runs from `runner.py`'s `finally:` block after the pipeline ends. Reads
today's daily log (user statements only), the current `people/<user>.md`,
and every `topics/*.md`, then asks the local LLM to re-emit any pages
that should change. Output is a JSON object `{relpath: new_markdown}`;
only `people/<active_user>.md` and `topics/<slug>.md` paths are accepted.
Each overwritten page is snapshotted to `wiki/.history/<date>/<path>`
before the new content lands, so a bad pass is always revertable.

Failure is non-fatal: Ollama down, malformed JSON, or a disallowed path
all log a warning and return without touching the wiki. The pipeline
exit path must never raise because of the librarian.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import httpx
from loguru import logger

from voiceassistant import config
from voiceassistant.memory.build_index import upsert_path
from voiceassistant.session import SessionContext
from voiceassistant.wiki.paths import wiki_dir
from voiceassistant.wiki.retriever import _user_statements_from_daily
from voiceassistant.wiki.store import read_page, write_page

_TOPIC_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _load_context(root: Path, user_id: str) -> dict[str, str]:
    """Gather the librarian's read-only inputs as a {relpath: body} map."""
    ctx: dict[str, str] = {}

    today = date.today().isoformat()
    daily = read_page(f"daily/{today}.md") or ""
    user_only = _user_statements_from_daily(daily).strip()
    if not user_only:
        return {}
    ctx[f"daily/{today}.md (user statements, today)"] = user_only

    person_rel = f"people/{user_id}.md"
    person = read_page(person_rel)
    ctx[person_rel + " (current)"] = (person or "").strip() or "(empty — create it)"

    topics_dir = root / "topics"
    if topics_dir.exists():
        for p in sorted(topics_dir.glob("*.md")):
            rel = f"topics/{p.name}"
            ctx[rel + " (current)"] = p.read_text(encoding="utf-8").strip()

    schema = read_page("schema.md")
    if schema:
        ctx["schema.md (rules)"] = schema.strip()

    return ctx


def _build_prompt(ctx: dict[str, str], user_id: str) -> str:
    parts = [
        "You are the end-of-session librarian for a personal voice assistant's wiki.",
        f"Today is {date.today().isoformat()}. The active user is '{user_id}'.",
        "",
        "HARD RULES — violating any of these is worse than returning {}:",
        "1. NEVER invent facts. Every statement you promote MUST be directly",
        "   supported by a verbatim substring of a user bullet in the input.",
        "   If you cannot quote the user saying it, do not write it.",
        "2. NEVER promote bot claims. The input already filters to user-only",
        "   lines; trust that filter and do not speculate about what the bot said.",
        "3. Recency wins: if a newer user statement contradicts an older fact on",
        "   an existing page, replace the older fact (don't keep both).",
        "4. Allowed output paths: 'people/" + user_id + ".md' and 'topics/<slug>.md'",
        "   (slug matches [a-z0-9][a-z0-9-]*). Any other path will be rejected.",
        "5. Re-emit each touched page in full. Preserve prior-day facts unchanged",
        "   unless today's user statements contradict them.",
        "6. For each promoted fact on a page, include the source quote inline",
        "   as a blockquote, e.g.:",
        "       ## Favorite color",
        "       Green.",
        "       > [HH:MM:SS] \"mi color favorito es verde\"",
        "7. Cross-link with [[topics/<slug>]] wikilinks from the person page to",
        "   topic pages; topic pages link back to [[people/" + user_id + "]].",
        "8. If no promotable user facts appeared today, return exactly {}.",
        "",
        "OUTPUT FORMAT — ONE JSON object, no prose outside it. Keys are",
        "wiki-relative paths; values are complete new Markdown content.",
        "",
        "--- inputs ---",
    ]
    for label, body in ctx.items():
        parts.append(f"\n### {label}\n{body}")
    return "\n".join(parts)


async def _call_ollama(prompt: str) -> dict[str, str] | None:
    """Call /api/chat with format=json. Returns parsed dict or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
    except Exception as e:
        logger.warning(f"librarian: Ollama call failed ({e})")
        return None

    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"librarian: malformed JSON from LLM ({e}); content={content[:200]!r}")
        return None

    if not isinstance(obj, dict):
        logger.warning(f"librarian: LLM returned non-object ({type(obj).__name__})")
        return None
    # Coerce all values to str; drop anything non-string.
    return {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, str)}


def _is_allowed_path(relpath: str, user_id: str) -> bool:
    if relpath == f"people/{user_id}.md":
        return True
    if relpath.startswith("topics/") and relpath.endswith(".md"):
        slug = relpath[len("topics/") : -len(".md")]
        if _TOPIC_SLUG.match(slug):
            return True
    return False


def _snapshot(root: Path, relpath: str) -> None:
    src = root / relpath
    if not src.exists():
        return
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    dst = root / ".history" / date.today().isoformat() / stamp / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


async def run(session: SessionContext, root: Path | None = None) -> None:
    """Best-effort librarian pass. Never raises."""
    try:
        root = root or wiki_dir()
        if not root.exists():
            return

        ctx = _load_context(root, session.user_id)
        if not ctx:
            logger.debug("librarian: no user statements today, skipping")
            return

        prompt = _build_prompt(ctx, session.user_id)
        patch = await _call_ollama(prompt)
        if not patch:
            return

        written: list[str] = []
        for relpath, new_content in patch.items():
            if not _is_allowed_path(relpath, session.user_id):
                logger.warning(f"librarian: refusing disallowed path '{relpath}'")
                continue
            if not new_content.strip():
                logger.debug(f"librarian: skipping empty content for '{relpath}'")
                continue
            current = read_page(relpath) or ""
            if current.strip() == new_content.strip():
                continue
            _snapshot(root, relpath)
            write_page(relpath, new_content if new_content.endswith("\n") else new_content + "\n")
            await upsert_path(root, relpath)
            written.append(relpath)

        if written:
            logger.info(f"librarian: updated {len(written)} page(s): {', '.join(written)}")
        else:
            logger.debug("librarian: no pages changed")
    except Exception as e:
        logger.warning(f"librarian: unexpected failure ({e}); session exit unaffected")
