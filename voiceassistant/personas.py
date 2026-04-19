"""Persona loading — voice + prompt + tool allowlist per persona_id.

Personas live in `wiki/personas/<persona_id>.md` with a loose YAML-ish
header (`---` fenced `key: value` pairs) followed by the system prompt as
free-form markdown. No new dependency — parsed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceassistant import config
from voiceassistant.wiki.store import read_page


@dataclass(frozen=True)
class Persona:
    persona_id: str
    system_prompt: str
    piper_voice: str
    tool_allowlist: tuple[str, ...] = ()
    language: str | None = None
    whisper_model: str | None = None
    whisper_compute_type: str | None = None


def _parse_persona_markdown(persona_id: str, text: str) -> Persona:
    lines = text.splitlines()
    meta: dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = i + 1
                break
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
    body = "\n".join(lines[body_start:]).strip()
    allow = tuple(
        s.strip() for s in meta.get("tool_allowlist", "").split(",") if s.strip()
    )
    return Persona(
        persona_id=persona_id,
        system_prompt=body,
        piper_voice=meta.get("voice", config.PIPER_VOICE_DEFAULT),
        tool_allowlist=allow,
        language=meta.get("language") or None,
        whisper_model=meta.get("whisper_model") or None,
        whisper_compute_type=meta.get("whisper_compute_type") or None,
    )


def load_persona(persona_id: str) -> Persona:
    text = read_page(f"personas/{persona_id}.md")
    if text is None:
        raise FileNotFoundError(
            f"Persona '{persona_id}' not found at wiki/personas/{persona_id}.md"
        )
    return _parse_persona_markdown(persona_id, text)
