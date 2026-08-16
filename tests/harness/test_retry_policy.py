from types import SimpleNamespace

import pytest

from harness.tools.retry_policy import (
    allocate_tool_attempts,
    should_publish_unavailable,
    tool_attempt_counts,
)
from harness.session.types import EventDraft
from harness.session.store import InMemorySessionStore


def test_allocate_blocks_third_attempt_same_tool():
    calls = [
        SimpleNamespace(name="lookup_fail", call_id="c1"),
        SimpleNamespace(name="lookup_fail", call_id="c2"),
        SimpleNamespace(name="lookup_fail", call_id="c3"),
        SimpleNamespace(name="submit_answer", call_id="s1"),
    ]
    blocked, counts = allocate_tool_attempts(calls, prior_counts={})
    assert blocked == {"c3"}
    assert counts["lookup_fail"] == 2
    assert "submit_answer" not in counts


def test_allocate_respects_prior_counts():
    calls = [SimpleNamespace(name="lookup_fail", call_id="c3")]
    blocked, _counts = allocate_tool_attempts(calls, prior_counts={"lookup_fail": 2})
    assert blocked == {"c3"}


@pytest.mark.asyncio
async def test_tool_attempt_counts_reads_turn_calls():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await store.append(
        header.session_id,
        EventDraft(event_type="tool/call", turn=1, data={"name": "lookup_fail", "call_id": "c1"}),
    )
    await store.append(
        header.session_id,
        EventDraft(event_type="tool/call", turn=1, data={"name": "submit_answer", "call_id": "s1"}),
    )
    events = await store.load_events(header.session_id)
    assert tool_attempt_counts(events, turn=1) == {"lookup_fail": 1}


def test_should_publish_unavailable_when_retry_exhausted_and_no_success():
    from harness.session.types import SessionEvent
    from datetime import datetime, timezone

    event = SessionEvent(
        seq=1,
        event_id="e",
        event_type="tool/result",
        schema_version=1,
        run_id="r",
        turn=1,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="append",
        source_event_seqs=(),
        visibility="public",
        data={"name": "lookup_fail", "ok": False, "content": "{\"error\":\"retry_exhausted\"}"},
        created_at=datetime.now(timezone.utc),
    )
    assert should_publish_unavailable(
        [{"ok": False, "error": "retry_exhausted"}],
        events=[event],
        turn=1,
    )
