from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from capabilities.analysis import normalize_materials, synthesize_answer
from harness.llm.fake import FakeLlmAdapter, content_turn, tool_turn
from harness.projection.sse import tool_family
from harness.session.store import InMemorySessionStore
from harness.session.types import EventDraft, SessionEvent
from harness.tools.analysis import collect_turn_materials
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.runtime import ToolRuntime
from harness.tools.skill import skill_definition
from skills.loader import list_skill_catalog


def _event(*, event_type: str, turn: int = 1, data: dict) -> SessionEvent:
    return SessionEvent(
        seq=1,
        event_id="e",
        event_type=event_type,
        schema_version=1,
        run_id="r",
        turn=turn,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="append",
        source_event_seqs=(),
        visibility="public",
        data=data,
        created_at=datetime.now(timezone.utc),
    )


def test_skill_catalog_includes_financial_analysis():
    names = {entry.name for entry in list_skill_catalog()}
    assert "financial-analysis" in names


def test_collect_turn_materials_keeps_evidence_id_and_skips_skill():
    events = [
        _event(
            event_type="tool/result",
            data={
                "name": "skill",
                "ok": True,
                "content": "instructions",
                "evidence_id": "skip-me",
            },
        ),
        _event(
            event_type="tool/result",
            data={
                "name": "lookup_ok",
                "ok": True,
                "content": "营收 100",
                "evidence_id": "ev-keep",
            },
        ),
        _event(
            event_type="tool/result",
            data={"name": "finalign_analyze", "ok": True, "content": "should skip"},
        ),
    ]
    materials = collect_turn_materials(events, turn=1)
    assert len(materials) == 1
    assert materials[0]["evidence_id"] == "ev-keep"
    assert materials[0]["content"] == "营收 100"


def test_normalize_materials_accepts_strings_and_dicts():
    items = normalize_materials(
        [
            "纯文本",
            {"content": "带编号", "evidence_id": "e2", "tool": "lookup_ok"},
            {"text": ""},
        ]
    )
    assert [item["content"] for item in items] == ["纯文本", "带编号"]
    assert items[1]["evidence_id"] == "e2"


@pytest.mark.asyncio
async def test_synthesize_answer_uses_finance_llm_and_stamps_evidence(monkeypatch):
    from app.core.config import settings

    class FakeLlm:
        async def ainvoke(self, messages):
            human = messages[1][1]
            assert "营收 100" in human
            assert "evidence_id=ev-keep" in human
            return SimpleNamespace(content="综合结论：营收 100 元")

    monkeypatch.setattr(settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr("agents.llm.is_finance_llm_available", lambda: True)
    monkeypatch.setattr("agents.llm.get_finance_llm", lambda: FakeLlm())

    result = await synthesize_answer(
        question="营收多少",
        materials=[{"content": "营收 100", "evidence_id": "ev-keep", "tool": "lookup_ok"}],
    )
    assert result["ok"] is True
    assert result["content"] == "综合结论：营收 100 元"
    assert result["evidence_ids"] == ["ev-keep"]
    assert str(result.get("evidence_id") or "").startswith("finalign.analyze:")


@pytest.mark.asyncio
async def test_synthesize_answer_does_not_fallback_to_deepseek(monkeypatch):
    from app.core.config import settings

    called = {"faq": False}

    def _faq():
        called["faq"] = True
        raise AssertionError("must not use DeepSeek as finalign fallback")

    monkeypatch.setattr(settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr("agents.llm.is_finance_llm_available", lambda: False)
    monkeypatch.setattr("agents.llm.get_faq_llm", _faq)
    monkeypatch.setattr("agents.llm.get_finance_llm", _faq)

    result = await synthesize_answer(
        question="营收多少",
        materials=[{"content": "营收 100", "evidence_id": "ev-keep"}],
    )
    assert result["ok"] is False
    assert result["error"] == "finalign_unavailable"
    assert called["faq"] is False


@pytest.mark.asyncio
async def test_synthesize_answer_requires_materials():
    result = await synthesize_answer(question="营收多少", materials=[])
    assert result["ok"] is False
    assert result["error"] == "no_evidence_to_analyze"


def test_sse_labels_finalign_as_analysis():
    assert tool_family("finalign_analyze") == "analysis"


@pytest.mark.asyncio
async def test_loop_analyze_reads_prior_tool_result(monkeypatch):
    from harness.agent.loop import Agent

    async def _lookup(_arguments: dict) -> dict:
        return {"ok": True, "content": "营收 100", "evidence_id": "ev-1"}

    async def _fake_synthesize(*, question, materials):
        assert question == "对比营收"
        assert any(item.get("evidence_id") == "ev-1" for item in materials)
        return {
            "ok": True,
            "content": "两家合计营收 100",
            "evidence_id": "finalign.analyze:test",
            "evidence_ids": ["ev-1"],
        }

    monkeypatch.setattr("harness.tools.analysis.synthesize_answer", _fake_synthesize)
    monkeypatch.setattr("harness.agent.loop.finalign_is_ready", lambda: True)

    lookup = ToolDefinition(
        tool_id="lookup.ok",
        name="lookup_ok",
        description="ok",
        handler=_lookup,
        openai_schema=function_schema("lookup_ok", "ok"),
        is_concurrency_safe=True,
    )
    analyze = ToolDefinition(
        tool_id="finalign.analyze",
        name="finalign_analyze",
        description="analyze",
        handler=_lookup,
        openai_schema=function_schema("finalign_analyze", "analyze"),
        is_concurrency_safe=False,
    )
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = FakeLlmAdapter(
        [
            tool_turn("lookup_ok", "{}", call_id="c1"),
            tool_turn("finalign_analyze", '{"question":"对比营收"}', call_id="c2"),
            content_turn("两家合计营收 100"),
        ]
    )
    agent = Agent(
        header.session_id,
        store,
        llm,
        runtime=ToolRuntime((skill_definition(), lookup, analyze)),
        owner_id="1",
    )
    result = await agent.prompt("对比营收")
    assert result.finish_reason == "completed"
    assert result.published_answer == "两家合计营收 100"
    results = [event for event in result.events if event.event_type == "tool/result"]
    assert results[0].data.get("evidence_id") == "ev-1"
    assert results[1].data.get("evidence_id") == "finalign.analyze:test"
    assert results[1].data.get("content") == "两家合计营收 100"


@pytest.mark.asyncio
async def test_loop_hides_finalign_and_answers_from_retrieval(monkeypatch):
    from harness.agent.loop import Agent
    from harness.tools.retry_policy import USER_UNAVAILABLE_HINT

    async def _lookup(_arguments: dict) -> dict:
        return {"ok": True, "content": "营收 100", "evidence_id": "ev-1"}

    monkeypatch.setattr("harness.agent.loop.finalign_is_ready", lambda: False)

    lookup = ToolDefinition(
        tool_id="lookup.ok",
        name="lookup_ok",
        description="ok",
        handler=_lookup,
        openai_schema=function_schema("lookup_ok", "ok"),
        is_concurrency_safe=True,
    )
    analyze = ToolDefinition(
        tool_id="finalign.analyze",
        name="finalign_analyze",
        description="analyze",
        handler=_lookup,
        openai_schema=function_schema("finalign_analyze", "analyze"),
        is_concurrency_safe=False,
    )
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    llm = FakeLlmAdapter(
        [
            tool_turn("lookup_ok", "{}", call_id="c1"),
            content_turn("根据查询，营收 100"),
        ]
    )
    agent = Agent(
        header.session_id,
        store,
        llm,
        runtime=ToolRuntime((skill_definition(), lookup, analyze)),
        owner_id="1",
    )
    result = await agent.prompt("对比营收")
    assert result.finish_reason == "completed"
    assert result.published_answer == "根据查询，营收 100"
    assert result.published_answer != USER_UNAVAILABLE_HINT
    tool_names = [
        str((item.get("function") or item).get("name") or "")
        for req in llm.requests
        for item in req["tools"]
        if isinstance(item, dict)
    ]
    assert "finalign_analyze" not in tool_names
    injected = [
        event
        for event in result.events
        if event.event_type == "user/message" and event.data.get("source") == "plugin"
    ]
    assert injected == []
