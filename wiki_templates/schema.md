# Schema

Rules the librarian follows when updating the wiki. MVP is append-only;
later phases let an LLM diff pages against these rules.

## Pages

- `personas/<persona_id>.md` — YAML-ish header (`voice`, `tool_allowlist`)
  followed by the persona's system prompt as free-form markdown.
- `devices/<device_id>.md` — YAML-ish header (`persona`) followed by
  human-readable device notes.
- `people/<user_id>.md` — free-form facts about the user. Stable
  preferences, recurring topics, relationships. Not a chat log.
- `daily/YYYY-MM-DD.md` — append-only turn blocks, newest at bottom.
- `log.md` — one line per turn for quick scan.

## Librarian rules (MVP)

- Never delete existing lines.
- Only append to `daily/<date>.md` and `log.md`.
- Turn blocks follow the exact format in `voiceassistant/wiki/librarian.py`.
- One write per turn per file.

## Retrieval rules (MVP)

- Every turn injects: active persona, active device, active user, last
  entry from today's daily log (if present).
- Cap total injected chars at `WIKI_INJECT_BUDGET_CHARS` (default 4000).
- Keyword-based page suggestion from `index.md` is a later phase.
