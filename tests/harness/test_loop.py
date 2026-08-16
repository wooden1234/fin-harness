from __future__ import annotations

import pytest

from harness.agent.loop import Agent
from harness.compaction.compact import maybe_compact
from harness.compaction.policy import CompactPolicy
from harness.finalization.submit import execute_submit_answer
from harness.llm.deepseek import to_langchain_messages
from harness.llm.fake import FakeLlmAdapter, submit_turn, tool_turn
from harness.projection.sse import project_session_event
from harness.session.store import InMemorySessionStore, assert_contiguous
from harness.session.surface import derive_messages, messages_for_llm, public_sse_events
from harness.session.types import EventDraft
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.runtime import ToolRuntime, submit_answer_definition
from harness.tools.skill import skill_definition


@pytest.mark.asyncio
async def test_derive_messages_keeps_tool_calls():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await store.append(
        header.session_id,
        EventDraft(event_type="turn/start", turn=1, data={"turn": 1}),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "查营收", "source": "user"},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="assistant/message",
            turn=1,
            step=1,
            surface_op="append",
            data={
                "content": "",
                "tool_calls": [
                    {"call_id": "c1", "name": "finance-query", "arguments": "{\"q\":\"营收\"}"}
                ],
            },
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            step=1,
            surface_op="append",
            data={"call_id": "c1", "name": "finance-query", "ok": True, "content": "100", "evidence_id": "e1"},
        ),
    )
    events = await store.load_events(header.session_id)
    assert_contiguous(events)
    messages = messages_for_llm(events)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["name"] == "finance-query"
    assert messages[2]["role"] == "tool"
    assert messages[2]["call_id"] == "c1"
    lc = to_langchain_messages("sys", messages)
    assistant = lc[2]
    assert assistant.tool_calls[0]["name"] == "finance-query"
    assert assistant.tool_calls[0]["id"] == "c1"


def test_submit_answer_rejects_ungrounded_numbers():
    result = execute_submit_answer(
        {"mode": "direct", "direct_answer": "营收 100 元"},
        events=[],
        turn=1,
    )
    assert result["ok"] is False
    assert result["error"] == "numbers_require_grounded"


def test_submit_answer_grounded_requires_evidence():
    result = execute_submit_answer(
        {
            "mode": "grounded",
            "statements": [{"text": "营收 100 元", "evidence_ids": ["missing"]}],
        },
        events=[],
        turn=1,
    )
    assert result["ok"] is False
    assert result["error"] == "missing_or_unknown_evidence"


def test_assistant_chunk_never_public():
    from harness.session.types import SessionEvent
    from datetime import datetime, timezone

    event = SessionEvent(
        seq=1,
        event_id="e",
        event_type="assistant/chunk",
        schema_version=1,
        run_id=None,
        turn=1,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="none",
        source_event_seqs=(),
        visibility="internal",
        data={"text": "secret"},
        created_at=datetime.now(timezone.utc),
    )
    assert public_sse_events([event]) == []
    assert project_session_event(event) == []


def _approval_runtime() -> ToolRuntime:
    async def _secret(arguments: dict) -> dict:
        return {"ok": True, "content": "secret-ok", "evidence_id": "ev-1"}

    secret = ToolDefinition(
        tool_id="secret.lookup",
        name="secret_lookup",
        description="needs approval",
        handler=_secret,
        openai_schema=function_schema("secret_lookup", "needs approval"),
        is_concurrency_safe=True,
        requires_human_approval=True,
    )
    return ToolRuntime((skill_definition(), submit_answer_definition(), secret))


@pytest.mark.asyncio
async def test_hitl_resume_keeps_tool_history_and_submits():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = FakeLlmAdapter(
        [
            tool_turn("secret_lookup", "{}", call_id="call-secret"),
            submit_turn("已完成查询。"),
        ]
    )
    agent = Agent(header.session_id, store, llm, runtime=_approval_runtime(), owner_id="1")
    first = await agent.prompt("查一下")
    assert first.waiting_approval is True
    assert first.approval_id
    calls = [event for event in first.events if event.event_type == "tool/call"]
    results = [event for event in first.events if event.event_type == "tool/result"]
    assert len(calls) == 1
    assert results == []
    second = await agent.resume_approval(first.approval_id, decision="allow")
    assert second.finish_reason == "completed"
    assert second.published_answer == "已完成查询。"
    events = await store.load_events(header.session_id)
    results = [event for event in events if event.event_type == "tool/result"]
    assert any(event.data.get("call_id") == "call-secret" for event in results)
    history = llm.requests[-1]["messages"]
    assistant = next(item for item in history if item.get("role") == "assistant" and item.get("tool_calls"))
    assert assistant["tool_calls"][0]["name"] == "secret_lookup"


@pytest.mark.asyncio
async def test_chat_submit_without_tools():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = FakeLlmAdapter([submit_turn("你好，我是小财。")])
    agent = Agent(header.session_id, store, llm, runtime=ToolRuntime.builtin(), owner_id="1")
    result = await agent.prompt("你好")
    assert result.finish_reason == "completed"
    assert result.published_answer == "你好，我是小财。"
    published = [event for event in result.events if event.event_type == "answer/published"]
    chunks = [event for event in result.events if event.event_type == "assistant/chunk"]
    assert published
    assert all(event.visibility == "internal" for event in chunks)


@pytest.mark.asyncio
async def test_compaction_replaces_tool_result_without_splitting_pair():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await store.append(header.session_id, EventDraft(event_type="turn/start", turn=1, data={"turn": 1}))
    await store.append(
        header.session_id,
        EventDraft(
            event_type="assistant/message",
            turn=1,
            surface_op="append",
            data={"content": "", "tool_calls": [{"call_id": "c1", "name": "web.search", "arguments": "{}"}]},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            surface_op="append",
            data={"call_id": "c1", "name": "web.search", "ok": True, "content": "X" * 200},
        ),
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        turn=1,
        run_id="r",
        token_limit=1,
        trigger="context-overflow",
        policy=CompactPolicy(prune_chars=20, head_chars=5, tail_chars=5),
    )
    events = await store.load_events(header.session_id)
    messages = derive_messages(events)
    tools = [item for item in messages if item.role == "tool"]
    assistants = [item for item in messages if item.role == "assistant"]
    assert len(assistants) == 1
    assert len(tools) == 1
    assert tools[0].call_id == "c1"
    assert "truncated" in tools[0].content


def test_tool_call_result_pairing_helper():
    calls = [{"call_id": "a"}, {"call_id": "b"}]
    results = [{"call_id": "a"}, {"call_id": "b"}]
    assert [item["call_id"] for item in calls] == [item["call_id"] for item in results]
