# Routing

Per-turn model routing rules consulted by `voiceassistant/routing/router.py`.
Each rule re-targets a single LLM turn; matches are evaluated top-to-bottom
and the **first match wins**. If no rule matches, the turn uses the default
spec supplied via `--llm` (or config.LLM_DEFAULT_SPEC).

## Format

One rule per non-empty, non-comment line:

```
<regex> -> <llm-spec>
```

- `<regex>` is a Python regular expression matched (case-insensitive,
  `re.search`) against the latest user message body.
- `<llm-spec>` uses the same grammar as `--llm`:
  - `local`, `ollama`, `ollama/<model>`
  - `openrouter/<model>` (e.g. `openrouter/anthropic/claude-sonnet-4.5`)
- Lines starting with `#` are comments and ignored.
- Whitespace around `->` is ignored.

## Examples

Uncomment and edit to taste. None of these are active by default.

```
# \bcode\b|\bpython\b|\brust\b -> openrouter/deepseek/deepseek-chat
# \bdream\b|\bstory\b          -> openrouter/anthropic/claude-sonnet-4.5
# \bweather\b                  -> ollama/llama3.1:8b-ctx4k
```

## Notes

- Routing runs for every user turn; keep the rule list short (<20 lines) to
  avoid wasting startup time re-parsing.
- OpenRouter rules are silently skipped if `OPENROUTER_API_KEY` is unset —
  the turn falls back to the default spec.
- Cross-backend switching works (ollama → openrouter and back) because both
  backends share Pipecat's OpenAI-compatible client.
