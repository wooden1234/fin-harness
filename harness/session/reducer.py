"""Deterministic Session state projection from the immutable event sequence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from harness.session.types import SessionEvent


@dataclass(frozen=True, slots=True)
class SessionState:
    current_run_id: str | None = None
    current_turn: int = 0
    current_step: int = 0
    status: str = "idle"
    pending_approval_id: str | None = None
    pending_tool_calls: tuple[str, ...] = field(default_factory=tuple)
    published_event_id: str | None = None
    latest_compaction_id: str | None = None
    latest_todos: tuple[dict, ...] = field(default_factory=tuple)
    last_event_seq: int = 0


def state_to_dict(state: SessionState) -> dict:
    """Serialize a reducer state without depending on a database model."""
    return {
        "current_run_id": state.current_run_id,
        "current_turn": state.current_turn,
        "current_step": state.current_step,
        "status": state.status,
        "pending_approval_id": state.pending_approval_id,
        "pending_tool_calls": list(state.pending_tool_calls),
        "published_event_id": state.published_event_id,
        "latest_compaction_id": state.latest_compaction_id,
        "latest_todos": list(state.latest_todos),
        "last_event_seq": state.last_event_seq,
    }


def state_from_dict(data: dict | None) -> SessionState:
    """Read persisted state defensively so older snapshots remain usable."""
    data = data or {}
    return SessionState(
        current_run_id=data.get("current_run_id"),
        current_turn=int(data.get("current_turn") or 0),
        current_step=int(data.get("current_step") or 0),
        status=str(data.get("status") or "idle"),
        pending_approval_id=data.get("pending_approval_id"),
        pending_tool_calls=tuple(str(item) for item in data.get("pending_tool_calls") or ()),
        published_event_id=data.get("published_event_id"),
        latest_compaction_id=data.get("latest_compaction_id"),
        latest_todos=tuple(dict(item) for item in data.get("latest_todos") or () if isinstance(item, dict)),
        last_event_seq=int(data.get("last_event_seq") or 0),
    )


def reduce_session(
    snapshot: SessionState | None,
    events: Iterable[SessionEvent],
) -> SessionState:
    state = snapshot or SessionState()
    pending = list(state.pending_tool_calls)
    for event in sorted(events, key=lambda item: item.seq):
        if event.seq <= state.last_event_seq:
            continue
        data = event.data or {}
        updates = {
            "last_event_seq": event.seq,
            "current_run_id": event.run_id or state.current_run_id,
            "current_turn": int(event.turn or state.current_turn),
            "current_step": int(event.step or state.current_step),
        }
        if event.event_type == "turn/start":
            updates["status"] = "running"
        elif event.event_type == "turn/end":
            updates["status"] = str(data.get("reason") or "completed")
        elif event.event_type == "tool/call":
            call_id = str(data.get("call_id") or "")
            if call_id and call_id not in pending:
                pending.append(call_id)
        elif event.event_type == "tool/result":
            call_id = str(data.get("call_id") or "")
            pending = [item for item in pending if item != call_id]
        elif event.event_type == "approval/asked":
            updates["status"] = "waiting_approval"
            updates["pending_approval_id"] = str(data.get("approval_id") or "") or None
        elif event.event_type == "approval/decided":
            updates["status"] = "running"
            updates["pending_approval_id"] = None
        elif event.event_type == "answer/published":
            updates["published_event_id"] = event.event_id
        elif event.event_type == "compaction/summary":
            updates["latest_compaction_id"] = event.event_id
        elif event.event_type == "todo/write":
            updates["latest_todos"] = tuple(
                dict(item) for item in data.get("todos") or [] if isinstance(item, dict)
            )
        state = replace(state, pending_tool_calls=tuple(pending), **updates)
    return state


__all__ = ["SessionState", "reduce_session", "state_from_dict", "state_to_dict"]
