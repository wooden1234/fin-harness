"""成稿后处理测试：finalign 优先，不可用则 DeepSeek 成稿2。"""

import pytest
from langchain_core.messages import AIMessage

from agents.main_deep_agent import drafting as drafting_module
from agents.main_deep_agent.contracts import MainAgentResponse
from agents.main_deep_agent.state import MainAgentProgressJournal
from agents.orchestrator.contracts import Evidence


def _grounded_json() -> str:
    return (
        '{"mode":"grounded","heading":"结论",'
        '"statements":[{"text":"营收同比+8%","evidence_ids":["main:abc"],'
        '"statement_type":"fact"}],'
        '"tables":[],"gaps":[],"follow_ups":[],'
        '"direct_answer":"","clarification":""}'
    )


def _journal_with_evidence() -> MainAgentProgressJournal:
    journal = MainAgentProgressJournal()
    journal.evidence["main:abc"] = Evidence(
        evidence_id="main:abc",
        content="营收同比+8%",
        source_type="tool",
        provider="test",
        title="财报",
        metadata={
            "display_text": "营收同比+8%",
            "facts": [{"metric": "营业收入累计[20241231]", "value": 1}],
        },
    )
    return journal


def test_normalize_draft_payload_accepts_finalign_near_schema() -> None:
    normalized = drafting_module._normalize_draft_payload({
        "mode": "grounded",
        "direct_answer": {"净利润预计增幅": "97.26%至130.14%"},
        "heading": "业绩预告",
        "statements": [
            "根据材料 evidence_id main:abc，净利润预计增长97.26%至130.14%。",
        ],
        "tables": [],
        "gaps": [],
        "follow_ups": [],
    })

    response = MainAgentResponse.model_validate(normalized)
    assert response.direct_answer == ""
    assert response.statements[0].evidence_ids == ["main:abc"]


def test_normalize_draft_payload_accepts_deepseek_aliases() -> None:
    normalized = drafting_module._normalize_draft_payload({
        "mode": "grounded",
        "statements": [{
            "evidence_id": "main:abc",
            "结论文本": "净利润预计增幅中枢为113.70%。",
            "type": "calculation",
        }],
        "tables": [{
            "title": "预计增幅",
            "columns": ["下限", "上限", "中枢"],
            "rows": [["97.26%", "130.14%", "113.70%"]],
            "evidence_id": "main:abc",
        }],
        "gaps": {"不确定性": "最终以正式半年报为准。"},
        "follow_ups": "查看正式半年报",
    })

    response = MainAgentResponse.model_validate(normalized)
    assert response.statements[0].text == "净利润预计增幅中枢为113.70%。"
    assert response.statements[0].evidence_ids == ["main:abc"]
    assert response.tables[0].rows[0].evidence_ids == ["main:abc"]
    assert response.gaps == ["最终以正式半年报为准。"]


@pytest.mark.asyncio
async def test_refine_falls_back_to_deepseek_when_finalign_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(drafting_module.settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr(drafting_module, "is_finance_llm_available", lambda: False)

    class FakeDeepSeek:
        async def ainvoke(self, _messages):
            return AIMessage(content=_grounded_json())

    monkeypatch.setattr(drafting_module, "get_faq_llm", lambda: FakeDeepSeek())
    original = MainAgentResponse(mode="direct", direct_answer="旧草稿")

    refined = await drafting_module.refine_main_response_with_finance_llm(
        original, _journal_with_evidence()
    )

    assert refined is not None
    assert refined.mode == "grounded"
    assert refined.statements[0].evidence_ids == ["main:abc"]


@pytest.mark.asyncio
async def test_refine_applies_finance_json_draft(monkeypatch) -> None:
    monkeypatch.setattr(drafting_module.settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr(drafting_module, "is_finance_llm_available", lambda: True)
    captured: list = []

    class FakeFinance:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return AIMessage(content=_grounded_json())

    class BoomDeepSeek:
        async def ainvoke(self, _messages):
            raise AssertionError("deepseek draft should not run when finalign succeeds")

    monkeypatch.setattr(drafting_module, "get_finance_llm", lambda: FakeFinance())
    monkeypatch.setattr(drafting_module, "get_faq_llm", lambda: BoomDeepSeek())
    original = MainAgentResponse(mode="direct", direct_answer="旧草稿")

    refined = await drafting_module.refine_main_response_with_finance_llm(
        original,
        _journal_with_evidence(),
        query="营收增长多少？请说明驱动因素",
    )

    assert refined is not None
    assert refined.mode == "grounded"
    assert refined.statements[0].text == "营收同比+8%"
    user_prompt = str(captured[1].content)
    assert "<question>" in user_prompt
    assert "驱动因素" in user_prompt
    assert "不得编造" in str(captured[0].content) or "gaps" in str(captured[0].content)


@pytest.mark.asyncio
async def test_refine_falls_back_to_deepseek_when_finalign_invalid_json(
    monkeypatch,
) -> None:
    monkeypatch.setattr(drafting_module.settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr(drafting_module, "is_finance_llm_available", lambda: True)

    class FakeFinance:
        async def ainvoke(self, _messages):
            return AIMessage(content="不是 JSON")

    class FakeDeepSeek:
        async def ainvoke(self, _messages):
            return AIMessage(content=_grounded_json())

    monkeypatch.setattr(drafting_module, "get_finance_llm", lambda: FakeFinance())
    monkeypatch.setattr(drafting_module, "get_faq_llm", lambda: FakeDeepSeek())
    original = MainAgentResponse(mode="direct", direct_answer="旧草稿")

    refined = await drafting_module.refine_main_response_with_finance_llm(
        original, _journal_with_evidence()
    )

    assert refined is not None
    assert refined.mode == "grounded"


@pytest.mark.asyncio
async def test_refine_keeps_original_when_both_drafts_fail(monkeypatch) -> None:
    monkeypatch.setattr(drafting_module.settings, "FINANCE_LLM_DRAFT_ENABLED", True)
    monkeypatch.setattr(drafting_module, "is_finance_llm_available", lambda: True)

    class FakeBad:
        async def ainvoke(self, _messages):
            return AIMessage(content="不是 JSON")

    monkeypatch.setattr(drafting_module, "get_finance_llm", lambda: FakeBad())
    monkeypatch.setattr(drafting_module, "get_faq_llm", lambda: FakeBad())
    response = MainAgentResponse(mode="direct", direct_answer="保留我")

    assert await drafting_module.refine_main_response_with_finance_llm(
        response, MainAgentProgressJournal()
    ) is response
