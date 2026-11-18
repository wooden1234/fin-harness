from __future__ import annotations

import pytest

from harness.control.services import AgentControl
from harness.prompt.preferences import PreferenceContext
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime


@pytest.mark.asyncio
async def test_preference_context_reloads_on_each_prompt(monkeypatch):
    calls = 0

    async def _load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return PreferenceContext(preferences={"response_language": "zh-CN"})
        return PreferenceContext(preferences={"response_language": "en-US"})

    monkeypatch.setattr("harness.control.services.load_preference_context", _load)
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    control = AgentControl(store, header.session_id)

    first, _ = await control.request_context(
        turn=1, run_id="run-1", base_runtime=ToolRuntime.builtin()
    )
    second, _ = await control.request_context(
        turn=1, run_id="run-1", base_runtime=ToolRuntime.builtin()
    )

    assert calls == 2
    assert "response_language=zh-CN" in first
    assert "response_language=en-US" in second
    assert control.effective_preferences("run-1") == {"response_language": "en-US"}

    control.finish_run("run-1")
    assert control.effective_preferences("run-1") == {}


@pytest.mark.asyncio
async def test_preference_reload_keeps_last_success_when_later_load_fails(monkeypatch):
    calls = 0

    async def _load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return PreferenceContext(preferences={"response_language": "en-US"})
        return PreferenceContext(memory_load_succeeded=False)

    monkeypatch.setattr("harness.control.services.load_preference_context", _load)
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    control = AgentControl(store, header.session_id)

    await control.request_context(
        turn=1, run_id="run-1", base_runtime=ToolRuntime.builtin()
    )
    second, _ = await control.request_context(
        turn=1, run_id="run-1", base_runtime=ToolRuntime.builtin()
    )

    assert calls == 3
    assert "response_language=en-US" in second
    assert control.effective_preferences("run-1") == {"response_language": "en-US"}


@pytest.mark.asyncio
async def test_preference_context_retries_initial_transient_failure(monkeypatch):
    calls = 0

    async def _load(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return PreferenceContext(memory_load_succeeded=False)
        return PreferenceContext(preferences={"response_language": "en-US"})

    monkeypatch.setattr("harness.control.services.load_preference_context", _load)
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    control = AgentControl(store, header.session_id)

    system, _ = await control.request_context(
        turn=1, run_id="run-retry", base_runtime=ToolRuntime.builtin()
    )

    assert calls == 2
    assert "response_language=en-US" in system
