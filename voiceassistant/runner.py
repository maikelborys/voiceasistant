"""CLI entrypoint.

Parses --transport/--user/--device/--persona, builds a SessionContext,
resolves a TransportBundle via the factory, hands it to build_pipeline(),
then runs the PipelineTask. Text lands in step 5, websocket in Phase 9.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner

from voiceassistant import config
from voiceassistant.personas import load_persona
from voiceassistant.pipeline import build_pipeline
from voiceassistant.session import SessionContext, TransportKind
from voiceassistant.transports import make_transport
from voiceassistant.wiki.paths import ensure_wiki_seeded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voiceassistant",
        description="PipecatAssistant — modular local voice agent.",
    )
    parser.add_argument(
        "--transport",
        choices=["text", "local_audio", "websocket"],
        default="local_audio",
        help="Input/output frontend (default: local_audio)",
    )
    parser.add_argument(
        "--user",
        default="maikel",
        help="User ID — picks wiki/people/<user>.md (default: maikel)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device ID — defaults to 'stdin' for text, 'laptop' for local_audio",
    )
    parser.add_argument(
        "--persona",
        default="default",
        help="Persona ID — picks voice + system prompt (default: default)",
    )
    return parser.parse_args()


def _default_device_for(transport: TransportKind) -> str:
    return {"text": "stdin", "local_audio": "laptop", "websocket": "websocket"}[transport]


async def async_main() -> None:
    args = _parse_args()

    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)

    wiki = ensure_wiki_seeded()
    logger.debug(f"wiki: {wiki}")

    device = args.device or _default_device_for(args.transport)
    session = SessionContext.new(
        transport_kind=args.transport,
        device_id=device,
        user_id=args.user,
        persona_id=args.persona,
    )
    logger.info(
        f"session {session.short_id}: starting — "
        f"transport={session.transport_kind} user={session.user_id} "
        f"device={session.device_id} persona={session.persona_id}"
    )

    persona = load_persona(session.persona_id)
    bundle = make_transport(session)
    task = build_pipeline(session, bundle, persona)

    logger.info(
        f"session {session.short_id}: ready — "
        f"user={session.user_id} device={session.device_id} persona={session.persona_id}"
    )
    if session.transport_kind == "local_audio":
        logger.info("Speak into your microphone.")
    await PipelineRunner().run(task)


def main() -> None:
    asyncio.run(async_main())
