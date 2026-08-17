"""Session 事件与草稿。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def new_id() -> str:
    return str(uuid4())


@dataclass(frozen=True, slots=True)
class EventDraft:
    event_type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = ""
    schema_version: int = 1
    run_id: str | None = None
    turn: int | None = None
    step: int | None = None
    causation_seq: int | None = None
    correlation_id: str | None = None
    surface_op: str = "none"
    source_event_seqs: tuple[int, ...] = ()
    visibility: str = ""


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    event_id: str
    event_type: str
    schema_version: int
    run_id: str | None
    turn: int | None
    step: int | None
    causation_seq: int | None
    correlation_id: str | None
    surface_op: str
    source_event_seqs: tuple[int, ...]
    visibility: str
    data: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn": self.turn,
            "step": self.step,
            "causation_seq": self.causation_seq,
            "correlation_id": self.correlation_id,
            "surface_op": self.surface_op,
            "source_event_seqs": list(self.source_event_seqs),
            "visibility": self.visibility,
            "data": dict(self.data),
            "created_at": self.created_at.isoformat(),
        }


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
