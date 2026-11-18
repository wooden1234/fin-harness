from __future__ import annotations

import json

import pytest

from harness.agent.loop import Agent
from harness.llm.fake import FakeLlmAdapter, ScriptedTurn, content_turn, tool_turn
from harness.llm.types import StreamChunk
from harness.projection.sse import project_session_event
from harness.session.store import InMemorySessionStore
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.runtime import ToolRuntime
from harness.tools.skill import skill_definition
from harness.tools.todo import TODO_PLAN_APPROVAL_MESSAGE


def _todos(*statuses: str) -> str:
    return json.dumps(
        {
            "todos": [
                {"content": "查营收", "status": statuses[0]},
                {"content": "查利润", "status": statuses[1]},
                {"content": "对照口径", "status": statuses[2]},
            ]
        },
        ensure_ascii=False,
    )


def _batch(todo_arguments: str) -> ScriptedTurn:
    return ScriptedTurn(
        (
            StreamChunk(
                kind="tool_call_delta",
                index=0,
                call_id="todo-1",
                name="todo_write",
                arguments_delta=todo_arguments,
            ),
            StreamChunk(
                kind="tool_call_delta",
                index=1,
                call_id="look-1",
                name="lookup_revenue",
                arguments_delta="{}",
                finish_reason="tool_calls",
            ),
            StreamChunk(
                kind="tool_call_delta",
                index=2,
                call_id="skill-1",
                name="skill",
                arguments_delta=json.dumps({"name": "faq-knowledge"}),
            ),
        )
    )


def _lookup_runtime() -> ToolRuntime:
    async def _lookup(_arguments: dict) -> dict:
        return {"ok": True, "content": "营收 100"}

    lookup = ToolDefinition(
        tool_id="lookup.revenue",
        name="lookup_revenue",
        description="查营收",
        handler=_lookup,
        openai_schema=function_schema("lookup_revenue", "查营收"),
        is_concurrency_safe=True,
    )
    return ToolRuntime((lookup, skill_definition()))


@pytest.mark.asyncio
async def test_first_todo_waits_and_siblings_still_run():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    agent = Agent(
        header.session_id,
        store,
        FakeLlmAdapter([
            _batch(_todos("pending", "pending", "pending")),
            content_turn("已查完。"),
        ]),
        runtime=_lookup_runtime(),
        owner_id="1",
    )
    first = await agent.prompt("对比三家公司")
    assert first.waiting_approval is True
    events = await store.load_events(header.session_id)
    assert not any(event.event_type == "todo/write" for event in events)
    assert not any(event.data.get("call_id") == "look-1" for event in events)
    asked = next(event for event in events if event.event_type == "approval/asked")
    assert asked.data.get("message") == TODO_PLAN_APPROVAL_MESSAGE
    assert [item["name"] for item in asked.data.get("pending_calls") or []] == [
        "lookup_revenue",
        "skill",
    ]
    second = await agent.resume_approval(first.approval_id or "", decision="allow")
    assert second.published_answer == "已查完。"
    events = await store.load_events(header.session_id)
    lookup = next(event for event in events if event.data.get("call_id") == "look-1" and event.event_type == "tool/result")
    assert lookup.data.get("ok") is True
    assert any(
        event.event_type == "user/message" and event.data.get("source") == "skill" for event in events
    )


@pytest.mark.asyncio
async def test_plan_request_does_not_query_before_approval():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    agent = Agent(
        header.session_id,
        store,
        FakeLlmAdapter(
            [
                tool_turn("lookup_revenue", "{}", call_id="look-1"),
                tool_turn("todo_write", _todos("pending", "pending", "pending"), call_id="todo-1"),
                content_turn("同意后再查。"),
            ]
        ),
        runtime=_lookup_runtime(),
        owner_id="1",
    )
    first = await agent.prompt("请分步完成，并先列出任务清单")
    assert first.waiting_approval is True
    events = await store.load_events(header.session_id)
    blocked = next(event for event in events if event.data.get("call_id") == "look-1" and event.event_type == "tool/result")
    assert "todo_plan_required" in str(blocked.data.get("content"))
    assert not any(event.event_type == "todo/write" for event in events)


@pytest.mark.asyncio
async def test_allow_writes_plan_and_later_update_does_not_ask_again():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    agent = Agent(
        header.session_id,
        store,
        FakeLlmAdapter(
            [
                tool_turn("todo_write", _todos("pending", "pending", "pending"), call_id="todo-1"),
                tool_turn("todo_write", _todos("in_progress", "pending", "pending"), call_id="todo-2"),
                content_turn("已按清单完成。"),
            ]
        ),
        owner_id="1",
    )
    first = await agent.prompt("对比三家公司")
    assert first.waiting_approval is True
    second = await agent.resume_approval(first.approval_id or "", decision="allow")
    assert second.finish_reason == "completed"
    assert second.published_answer == "已按清单完成。"
    events = await store.load_events(header.session_id)
    assert sum(1 for event in events if event.event_type == "approval/asked") == 1
    assert sum(1 for event in events if event.event_type == "todo/write") == 2


@pytest.mark.asyncio
async def test_deny_blocks_another_todo_and_still_answers():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    agent = Agent(
        header.session_id,
        store,
        FakeLlmAdapter(
            [
                tool_turn("todo_write", _todos("pending", "pending", "pending"), call_id="todo-1"),
                tool_turn("todo_write", _todos("pending", "pending", "pending"), call_id="todo-2"),
                content_turn("直接完成。"),
            ]
        ),
        owner_id="1",
    )
    first = await agent.prompt("对比三家公司")
    second = await agent.resume_approval(first.approval_id or "", decision="deny")
    assert second.waiting_approval is False
    assert second.finish_reason == "completed"
    assert second.published_answer == "直接完成。"
    events = await store.load_events(header.session_id)
    assert not any(event.event_type == "todo/write" for event in events)
    assert sum(1 for event in events if event.event_type == "approval/asked") == 1
    declined = [
        event
        for event in events
        if event.event_type == "tool/result" and "todo_plan_declined" in str(event.data.get("content"))
    ]
    assert len(declined) == 2
