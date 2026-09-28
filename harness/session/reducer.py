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


__all__ = ["SessionState", "reduce_session"]
