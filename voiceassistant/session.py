"""SessionContext — per-run identity passed to every transport and processor.

Identity lives at the transport layer, not inside the LLM. One SessionContext
is created at startup (text/local_audio) or per-connection (websocket, future)
and threaded through the pipeline so downstream processors can route/log by
session, user, device, or persona without the LLM having to decide.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

TransportKind = Literal["text", "local_audio", "websocket"]


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    device_id: str
    user_id: str
    persona_id: str
    transport_kind: TransportKind
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def new(
        cls,
        *,
        transport_kind: TransportKind,
        device_id: str,
        user_id: str,
        persona_id: str,
    ) -> "SessionContext":
        return cls(
            session_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            persona_id=persona_id,
            transport_kind=transport_kind,
        )

    @property
    def short_id(self) -> str:
        return self.session_id[:8]
