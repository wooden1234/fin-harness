from __future__ import annotations

from datetime import datetime, timezone

from harness.session.reducer import SessionState, reduce_session, state_from_dict, state_to_dict
from harness.session.types import SessionEvent


def _event(seq: int, event_type: str, data: dict | None = None) -> SessionEvent:
    return SessionEvent(
        seq=seq,
        event_id=f"event-{seq}",
        event_type=event_type,
        schema_version=1,
        run_id="run-1",
        turn=1,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="none",
        source_event_seqs=(),
        visibility="internal",
        data=data or {},
        created_at=datetime.now(timezone.utc),
    )


def test_reducer_tracks_tool_and_approval_state() -> None:
    state = reduce_session(None, [
        _event(1, "turn/start"),
        _event(2, "tool/call", {"call_id": "call-1"}),
        _event(3, "approval/asked", {"approval_id": "approval-1"}),
    ])
    assert state.status == "waiting_approval"
    assert state.pending_tool_calls == ("call-1",)
    assert state.pending_approval_id == "approval-1"

    resumed = reduce_session(state, [
        _event(3, "approval/asked", {"approval_id": "approval-1"}),
        _event(4, "approval/decided", {"approval_id": "approval-1", "decision": "allow"}),
        _event(5, "tool/result", {"call_id": "call-1", "ok": True}),
        _event(6, "answer/published", {"markdown": "done"}),
        _event(7, "turn/end", {"reason": "completed"}),
    ])
    assert resumed.status == "completed"
    assert resumed.pending_tool_calls == ()
    assert resumed.pending_approval_id is None
    assert resumed.published_event_id == "event-6"
    assert resumed.last_event_seq == 7


def test_reducer_is_idempotent_for_already_applied_events() -> None:
    events = [_event(1, "turn/start"), _event(2, "turn/end", {"reason": "completed"})]
    first = reduce_session(None, events)
    assert reduce_session(first, events) == first


def test_reducer_snapshot_round_trip_and_tail_replay() -> None:
    initial = reduce_session(None, [_event(1, "turn/start"), _event(2, "tool/call", {"call_id": "c1"})])
    restored = state_from_dict(state_to_dict(initial))
    assert restored == initial

    resumed = reduce_session(restored, [_event(2, "tool/call", {"call_id": "c1"}), _event(3, "tool/result", {"call_id": "c1"})])
    assert resumed.last_event_seq == 3
    assert resumed.pending_tool_calls == ()
