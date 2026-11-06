from __future__ import annotations

import asyncio

import pytest

from harness.agent.loop import Agent
from harness.contracts.errors import LlmError
from harness.llm.deepseek import DeepSeekAdapter
from harness.llm.fake import FakeLlmAdapter
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_call_chunks = []


class _ScriptedChat:
    def __init__(self, chunks: list[_Chunk], *, hang: bool = False) -> None:
        self._chunks = chunks
        self._hang = hang

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages):
        if self._hang:
            await asyncio.Event().wait()
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_silent_model_stream_ends_the_turn(monkeypatch):
    monkeypatch.setattr("harness.llm.deepseek._stream_idle_timeout_seconds", lambda: 0.05)
    monkeypatch.setattr("agents.llm.get_faq_llm", lambda: _ScriptedChat([], hang=True))

    adapter = DeepSeekAdapter()
    with pytest.raises(LlmError) as caught:
        async for _chunk in adapter.stream(
            system="s",
            messages=[],
            tools=[],
            abort=asyncio.Event(),
        ):
            pass

    assert caught.value.code == "timeout"


@pytest.mark.asyncio
async def test_finished_stream_still_yields_text(monkeypatch):
    monkeypatch.setattr("agents.llm.get_faq_llm", lambda: _ScriptedChat([_Chunk("你好")]))
    adapter = DeepSeekAdapter()
    chunks = [
        chunk
        async for chunk in adapter.stream(
            system="s",
            messages=[],
            tools=[],
            abort=asyncio.Event(),
        )
    ]
    assert [chunk.text for chunk in chunks if chunk.kind == "content"] == ["你好"]
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_timeout_error_finishes_the_agent_turn():
    class _TimeoutLlm(FakeLlmAdapter):
        async def stream(self, *, system, messages, tools, abort):
            raise LlmError("模型响应超时", code="timeout", retryable=True)
            yield  # pragma: no cover

    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    agent = Agent(
        header.session_id,
        store,
        _TimeoutLlm(),
        runtime=ToolRuntime.builtin(),
        owner_id="1",
    )
    result = await agent.prompt("你好")
    assert result.finish_reason == "llm_timeout"
    assert result.error == "llm_timeout"
    assert result.published_answer is None


def test_deepseek_client_timeout_is_below_sdk_default(monkeypatch):
    from app.core.config import settings
    from agents.llm import _build_deepseek_llm

    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "DEEPSEEK_TIMEOUT_SEC", 60.0)
    monkeypatch.setattr(settings, "DEEPSEEK_STREAM_CHUNK_TIMEOUT_SEC", 30.0)
    llm = _build_deepseek_llm(temperature=0)
    read = getattr(llm.request_timeout, "read", llm.request_timeout)
    assert float(read) == 60.0
    assert float(read) < 600
    assert llm.stream_chunk_timeout == 30.0
