from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.finance_agent.pdf_agent.query_rewrite.answer_mismatch import node as mismatch_node


class _FakeLlm:
    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, messages, config=None):
        del messages, config
        return SimpleNamespace(content=self.content)


def test_build_rewrite_contract_extracts_exclusion_and_anchor():
    contract = mismatch_node.build_rewrite_contract(
        "寒武纪2024年营业收入是多少，不要答成龙芯"
    )

    assert contract.base_query == "寒武纪2024年营业收入"
    assert contract.must_not == ("龙芯",)
    assert contract.anchor == "年度报告 主要财务指标"


def test_forbidden_subject_variant_cannot_bypass_validation():
    contract = mismatch_node.build_rewrite_contract(
        "宁德时代2025年营业收入，不要用腾讯收入"
    )

    valid, reason = mismatch_node.validate_rewrite_query(
        "宁德时代2025年营业收入 对比腾讯营业收入 年度报告", contract
    )

    assert valid is False
    assert reason == "forbidden_term"


@pytest.mark.asyncio
async def test_answer_mismatch_accepts_valid_structured_rewrite(monkeypatch):
    monkeypatch.setattr(
        mismatch_node,
        "get_pdf_llm",
        lambda: _FakeLlm(
            '{"query":"寒武纪2024年营业收入 年度报告","confidence":0.91}'
        ),
    )

    result = await mismatch_node.answer_mismatch_node(
        {
            "original_query": "寒武纪2024年营业收入是多少，不要答成龙芯",
            "context": "龙芯中科2024年营业收入。",
        }
    )

    assert result["query"] == "寒武纪2024年营业收入 年度报告"
    assert "龙芯" not in result["query"]
    assert result["rewrite_reason"] == "question_context_mismatch"
    stage = result["rag_trace"]["stages"][-1]
    assert stage["status"] == "ok"
    assert stage["validation_reason"] == "ok"


@pytest.mark.asyncio
async def test_answer_mismatch_rejects_forbidden_term_and_uses_safe_fallback(monkeypatch):
    monkeypatch.setattr(
        mismatch_node,
        "get_pdf_llm",
        lambda: _FakeLlm(
            '{"query":"寒武纪2024年营业收入 对比龙芯 年度报告","confidence":0.9}'
        ),
    )

    result = await mismatch_node.answer_mismatch_node(
        {"original_query": "寒武纪2024年营业收入是多少，不要答成龙芯"}
    )

    assert result["query"] == "寒武纪2024年营业收入 年度报告 主要财务指标"
    assert "龙芯" not in result["query"]
    assert result["rewrite_reason"] == "question_context_mismatch_fallback"
    stage = result["rag_trace"]["stages"][-1]
    assert stage["status"] == "validated_fallback"
    assert stage["validation_reason"] == "forbidden_term"


@pytest.mark.asyncio
async def test_answer_mismatch_falls_back_when_required_anchor_is_missing(monkeypatch):
    monkeypatch.setattr(
        mismatch_node,
        "get_pdf_llm",
        lambda: _FakeLlm('{"query":"数字治理指数北京2024年数值","confidence":0.8}'),
    )

    result = await mismatch_node.answer_mismatch_node(
        {"original_query": "数字治理指数北京2024年数值，不是数字经济总指数"}
    )

    assert result["query"] == "数字治理指数北京2024年数值 分指数表"
    assert "数字经济总指数" not in result["query"]
    assert result["rag_trace"]["stages"][-1]["validation_reason"] == "missing_anchor"
