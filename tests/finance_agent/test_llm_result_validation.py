"""text_to_sql LLM 结果质检单元测试。"""

import pytest

from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result import (
    LlmResultValidationDecision,
    validate_query_result_with_llm,
)
from agents.finance_agent.financial_query_agent.text_to_sql.validation.result import (
    validate_query_result_full,
)

_FACT_SQL = """
SELECT company.name AS company_name, fact.raw_value AS raw_value
FROM fin_core.annual_financial_facts AS fact
LIMIT :limit
"""


class _FakeStructuredLlm:
    def __init__(self, decision: LlmResultValidationDecision):
        self._decision = decision

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages, config=None):
        return self._decision


@pytest.mark.asyncio
async def test_validate_query_result_with_llm_skips_when_disabled(monkeypatch):
    called = {"llm": False}

    def fake_get_router_llm():
        called["llm"] = True
        raise AssertionError("LLM should not be called when disabled")

    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        fake_get_router_llm,
    )

    result = await validate_query_result_with_llm(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="五粮液",
                metric_name="营业收入",
                raw_value="100",
            )
        ],
    )

    assert result.ok
    assert called["llm"] is False


@pytest.mark.asyncio
async def test_validate_query_result_with_llm_flags_wrong_sql(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        lambda: _FakeStructuredLlm(
            LlmResultValidationDecision(
                verdict="wrong_sql",
                reason="用户问营收，结果返回了毛利率。",
            )
        ),
    )

    result = await validate_query_result_with_llm(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="CATL",
                metric_name="毛利率",
                raw_value="20%",
            )
        ],
    )

    assert not result.ok
    assert result.error_type == "semantic"
    assert result.should_clarify is False
    assert "毛利率" in result.error


@pytest.mark.asyncio
async def test_validate_query_result_with_llm_flags_need_clarify(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        lambda: _FakeStructuredLlm(
            LlmResultValidationDecision(
                verdict="need_clarify",
                reason="“宁德”可能对应多家公司，请用户确认。",
            )
        ),
    )

    result = await validate_query_result_with_llm(
        question="宁德营业收入是多少",
        sql=_FACT_SQL,
        rows=[FinancialSqlResultRow(company_name="CATL", metric_name="收入", raw_value="100")],
    )

    assert not result.ok
    assert result.should_clarify is True


@pytest.mark.asyncio
async def test_validate_query_result_with_llm_fail_open_on_exception(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: True,
    )

    def boom():
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        boom,
    )

    result = await validate_query_result_with_llm(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[FinancialSqlResultRow(company_name="CATL", metric_name="收入", raw_value="100")],
    )

    assert result.ok


@pytest.mark.asyncio
async def test_validate_query_result_full_runs_llm_after_rules_pass(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        lambda: _FakeStructuredLlm(
            LlmResultValidationDecision(verdict="ok", reason="结果匹配问题。")
        ),
    )

    result = await validate_query_result_full(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="CATL",
                metric_name="营业收入",
                raw_value="4,000,000",
            )
        ],
    )

    assert result.ok


@pytest.mark.asyncio
async def test_validate_query_result_full_skips_llm_when_rules_fail(monkeypatch):
    called = {"llm": False}

    def fake_get_router_llm():
        called["llm"] = True
        raise AssertionError("rules failed, LLM should not run")

    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.is_llm_result_validation_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result.get_router_llm",
        fake_get_router_llm,
    )

    result = await validate_query_result_full(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[],
    )

    assert not result.ok
    assert result.error_type == "result_empty"
    assert called["llm"] is False
