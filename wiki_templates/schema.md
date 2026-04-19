# Schema

Rules the librarian follows when updating the wiki. Two librarians run:

1. **Per-turn (non-LLM, append-only)** — `voiceassistant/wiki/librarian.py`
   writes `daily/<date>.md` + `log.md`.
2. **End-of-session (LLM-driven)** — `voiceassistant/memory/librarian_llm.py`
   reads today's daily log + existing structured pages and re-emits whole
   pages under `people/` and `topics/` following the promotion rules below.

## Page types

- `personas/<persona_id>.md` — YAML-ish header (`voice`, `tool_allowlist`)
  followed by the persona's system prompt. **Librarian never edits.**
- `devices/<device_id>.md` — YAML-ish header (`persona`) followed by
  human-readable device notes. **Librarian never edits.**
- `people/<user_id>.md` — stable facts about one user: preferences,
  routines, recurring topics, relationships. Not a chat log. **Librarian
  may rewrite; previous version snapshotted to `.history/`.**
- `topics/<slug>.md` — one page per recurring topic the user talks about
  (e.g. `topics/coffee.md`). **Librarian may create and rewrite;
  snapshotted on edit.**
- `notes/<date>.md` — ad-hoc user-dictated notes via the `save_note` tool
  (Phase 3). Append-only. **Librarian does not edit.**
- `daily/<date>.md` — append-only raw turn log. **Immutable; librarian
  reads but never writes.**
- `log.md` — one-line-per-turn audit trail. **Immutable; librarian reads
  but never writes.**
- `index.md`, `schema.md`, `routing.md` — wiki meta-documents.
  **Librarian does not edit.**

## Per-turn librarian rules

- Never delete existing lines.
- Only append to `daily/<date>.md` and `log.md`.
- Turn blocks follow the format in `voiceassistant/wiki/librarian.py`.
- One write per turn per file.

## Memory promotion rules (end-of-session librarian)

**Input to the librarian prompt:**
- Today's `daily/<date>.md` filtered to user statements only (no bot lines).
- Current `people/<user>.md` for the active user.
- All current `topics/*.md` pages.
- This file (`schema.md`).

**Output contract:** a JSON object where keys are wiki-relative paths and
values are the complete new Markdown content of that file. Paths not
mentioned are left untouched.

**Path allowlist (librarian MUST refuse anything else):**
- `people/<active_user_id>.md` — exactly one path, matching the session's user.
- `topics/<slug>.md` — slug matches `[a-z0-9][a-z0-9-]*`.

**Content rules:**
1. **User-stated facts only.** Never promote a claim the bot made. If the
   user didn't say it today or in an earlier daily log, it doesn't go in.
2. **Recency wins.** When a new user statement contradicts an older fact,
   replace the older fact — don't keep both. Record the resolution date
   if helpful (`_as of 2026-04-19_`).
3. **Wikilinks.** When `people/<user>.md` mentions a topic that has (or
   should have) a topic page, link it as `[[topics/<slug>]]`. Same
   direction from topic pages back to people.
4. **Tags.** Use `#hashtags` sparingly for Obsidian graph navigation.
5. **No chat transcripts.** `people/` and `topics/` pages summarise;
   they never quote turn-by-turn dialogue. The daily log is the transcript.
6. **Immutability.** Never emit a path starting with `daily/`, `notes/`,
   `personas/`, `devices/`, or the names `log.md`, `index.md`, `schema.md`,
   `routing.md`.
7. **Small diffs preferred.** Don't rewrite everything on the page; change
   only what today's statements require. Structure, headings, and
   prior-day facts stay unless contradicted.
8. **Self-consistency.** A topic page mentioned from `people/<user>.md`
   should exist in the same output (create it if needed).

**Safety net:** every overwrite is snapshotted to
`wiki/.history/<YYYY-MM-DD>/<relpath>` before the new content is written,
so a bad librarian pass is always revertable.

## Retrieval rules

- Every turn runs semantic top-K over the vector index (sqlite-vec +
  nomic-embed-text).
- Pinned pages always injected regardless of score:
  `personas/<active>.md`, `devices/<active>.md`, `people/<user>.md`,
  `daily/<today>.md` (user statements only).
- Budget: `WIKI_INJECT_BUDGET_CHARS` chars total. Oversize semantic hits
  are skipped; pinned pages are truncated with `...[truncated]` rather
  than dropped.
- Cold path (empty index): fall back to the four pinned pages.
