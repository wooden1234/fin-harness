from __future__ import annotations

from typing import Any, AsyncIterator, Mapping, Sequence

import pytest
from langsmith.run_helpers import get_current_run_tree, tracing_context

from harness.agent.loop import Agent
from harness.llm.fake import FakeLlmAdapter, content_turn, tool_turn
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime
from harness.tracing import (
    current_trace_headers,
    distributed_parent,
    incoming_trace_context,
    thread_metadata,
    turn_langsmith_extra,
)


class CapturingLlm(FakeLlmAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trees: list[Any] = []

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        abort: Any,
    ) -> AsyncIterator[Any]:
        self.trees.append(get_current_run_tree())
        async for chunk in super().stream(
            system=system, messages=messages, tools=tools, abort=abort
        ):
            yield chunk


def _tree_meta(tree: Any) -> dict[str, Any]:
    extra = getattr(tree, "extra", None) or {}
    if isinstance(extra, dict) and isinstance(extra.get("metadata"), dict):
        return dict(extra["metadata"])
    return dict(getattr(tree, "metadata", None) or {})


@pytest.mark.asyncio
async def test_prompt_opens_parent_span_without_manual_id():
    extra = turn_langsmith_extra("sess-1", entry="prompt")
    assert "parent" not in extra
    assert "run_id" not in extra
    assert extra["metadata"]["thread_id"] == "sess-1"

    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = CapturingLlm([content_turn("你好")])
    agent = Agent(header.session_id, store, llm, runtime=ToolRuntime.builtin(), owner_id="1")

    with tracing_context(enabled="local"):
        result = await agent.prompt("上海天气")

    assert result.finish_reason == "completed"
    assert llm.trees and llm.trees[0] is not None
    parent = llm.trees[0]
    assert parent.name == "agent.turn"
    assert parent.parent_run_id is None
    assert _tree_meta(parent)["thread_id"] == header.session_id


@pytest.mark.asyncio
async def test_model_and_tools_share_parent_span_not_thread_id():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = CapturingLlm(
        [
            tool_turn("skill", '{"name":"missing"}', call_id="c1"),
            content_turn("没找到技能"),
        ]
    )
    agent = Agent(header.session_id, store, llm, runtime=ToolRuntime.builtin(), owner_id="1")

    with tracing_context(enabled="local"):
        result = await agent.prompt("查一下")

    assert result.finish_reason == "completed"
    assert len(llm.trees) == 2
    first, second = llm.trees
    assert first is not None and second is not None
    assert first.id == second.id
    assert first.name == "agent.turn"
    assert _tree_meta(first)["thread_id"] == header.session_id
    assert first.id != header.session_id


@pytest.mark.asyncio
async def test_follow_up_reuses_thread_id_but_new_parent_span():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = CapturingLlm([content_turn("第一轮"), content_turn("追问")])
    agent = Agent(header.session_id, store, llm, runtime=ToolRuntime.builtin(), owner_id="1")

    with tracing_context(enabled="local"):
        await agent.prompt("先问")
        await agent.prompt("再问")

    assert len(llm.trees) == 2
    first, second = llm.trees
    assert _tree_meta(first)["thread_id"] == _tree_meta(second)["thread_id"] == header.session_id
    assert first.id != second.id


def test_distributed_parent_reads_headers_only():
    assert distributed_parent({}) is None
    assert distributed_parent({"x-request-id": "abc"}) is None
    headers = {"langsmith-trace": "dotted", "baggage": "k=v", "authorization": "nope"}
    assert distributed_parent(headers) == {
        "langsmith-trace": "dotted",
        "baggage": "k=v",
    }


def test_thread_metadata_is_not_a_parent_id():
    meta = thread_metadata("conv-9")
    assert meta == {"thread_id": "conv-9", "session_id": "conv-9"}


@pytest.mark.asyncio
async def test_incoming_headers_become_parent_without_inventing_id():
    captured: dict[str, Any] = {}

    from langsmith import traceable

    @traceable(name="downstream", run_type="chain")
    async def downstream() -> None:
        tree = get_current_run_tree()
        captured["tree"] = tree
        captured["headers"] = current_trace_headers()

    with tracing_context(enabled="local"):
        @traceable(name="upstream", run_type="chain")
        async def upstream() -> None:
            headers = current_trace_headers()
            assert "langsmith-trace" in headers
            with incoming_trace_context(headers, thread_id="thread-a"):
                await downstream()

        await upstream()

    tree = captured["tree"]
    assert tree is not None
    assert tree.name == "downstream"
    assert tree.parent_run_id is not None
    assert _tree_meta(tree)["thread_id"] == "thread-a"
