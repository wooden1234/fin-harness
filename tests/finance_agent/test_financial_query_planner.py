"""financial_query_agent 内部 planner 路由契约测试。"""

from __future__ import annotations

from typing import get_args, get_type_hints
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.finance_agent.financial_query_agent.predefined.extraction.models import (
    PredefinedSlotExtraction,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection.node import (
    PredefinedToolSelectionResult,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection.prompts import (
    PREDEFINED_TOOL_SELECTION_PROMPT,
)
from app.agents.finance_agent.financial_query_agent.planner.node import (
    _looks_like_complex_query,
    financial_query_planner,
)
from agents.finance_agent.financial_query_agent.state import (
    FinancialQueryCoverage,
    FinancialQueryNextActionSql,
    FinancialQueryRoute,
    FinancialQueryState,
    FinancialQueryValidationErrorType,
)


def test_looks_like_complex_query_detects_ranking_keywords():
    assert _looks_like_complex_query("2024年营收排名前十的公司") is True
    assert _looks_like_complex_query("宁德时代2024年营业收入") is False


def test_financial_query_state_contract_centralizes_routes_and_validation_fields():
    fields = get_type_hints(FinancialQueryState)

    assert "financial_query_validation_error_type" in fields
    assert "financial_query_validation_error_types" in fields
    assert get_args(FinancialQueryRoute) == ("predefined", "text_to_sql")
    assert get_args(FinancialQueryCoverage) == ("covered", "clarify", "uncovered")
    assert "fallback_to_text_to_sql" in get_args(FinancialQueryNextActionSql)
    assert "result_schema" in get_args(FinancialQueryValidationErrorType)


def test_tool_selection_prompt_is_compact_and_keeps_contracts():
    assert len(PREDEFINED_TOOL_SELECTION_PROMPT) <= 1600
    for template_id in (
        "exact_metric_lookup",
        "latest_metric_lookup",
        "compare_metric_lookup",
        "compare_year_metric_lookup",
        "trend_metric_lookup",
    ):
        assert PREDEFINED_TOOL_SELECTION_PROMPT.count(template_id) == 1

    for boundary in ("季度", "同比", "占比", "多指标", "具体年份"):
        assert boundary in PREDEFINED_TOOL_SELECTION_PROMPT


@pytest.mark.asyncio
async def test_planner_routes_complex_question_to_text_to_sql_without_llm():
    state = {
        "messages": [HumanMessage(content="2024年营收排名前十的公司")],
        "sub_question": "2024年营收排名前十的公司",
    }

    with patch(
        "app.agents.finance_agent.financial_query_agent.planner.node.select_predefined_tool",
        new=AsyncMock(),
    ) as mock_select_tool:
        result = await financial_query_planner(state)

    mock_select_tool.assert_not_awaited()
    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["steps"] == ["financial_query_planner"]


@pytest.mark.asyncio
async def test_planner_selects_template_and_slots_in_one_llm_call():
    state = {
        "messages": [HumanMessage(content="腾讯2024年营业收入")],
        "sub_question": "腾讯2024年营业收入",
    }
    selection = PredefinedToolSelectionResult(
        success=True,
        template_id="exact_metric_lookup",
        slots=PredefinedSlotExtraction(
            companies=["腾讯"],
            years=[2024],
            metrics=["营业收入"],
            operation="lookup",
        ),
    )

    with patch(
        "app.agents.finance_agent.financial_query_agent.planner.node.select_predefined_tool",
        new=AsyncMock(return_value=selection),
    ) as mock_select_tool:
        result = await financial_query_planner(state)

    mock_select_tool.assert_awaited_once()
    assert result["financial_query_plan_route"] == "predefined"
    assert result["financial_query_template_id"] == "exact_metric_lookup"
    assert result["financial_query_intent"].companies == ["Tencent"]


@pytest.mark.asyncio
async def test_planner_routes_failed_template_selection_to_text_to_sql():
    state = {
        "messages": [HumanMessage(content="腾讯的某项自定义经营指标")],
        "sub_question": "腾讯的某项自定义经营指标",
    }
    selection = PredefinedToolSelectionResult(
        success=False,
        template_id="",
        slots=None,
        error="tool_selection_missing_tool_call",
    )

    with patch(
        "app.agents.finance_agent.financial_query_agent.planner.node.select_predefined_tool",
        new=AsyncMock(return_value=selection),
    ):
        result = await financial_query_planner(state)

    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["financial_query_plan_reason"] == "tool_selection_missing_tool_call"
    assert "financial_query_intent" not in result


@pytest.mark.asyncio
async def test_planner_empty_question_returns_database_failure():
    state = {"messages": [], "sub_question": ""}

    result = await financial_query_planner(state)

    assert result["steps"] == ["financial_query_planner_error"]
    assert result["messages"] == []
    assert result["task_results"][0]["coverage"] == "uncovered"
