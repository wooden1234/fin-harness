"""financial_query_agent 内部 planner 路由契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents.finance_agent.financial_query_agent.planner.models import (
    FinancialQueryPlan,
)
from app.agents.finance_agent.financial_query_agent.planner.node import (
    _looks_like_complex_query,
    financial_query_planner,
)


def test_looks_like_complex_query_detects_ranking_keywords():
    assert _looks_like_complex_query("2024年营收排名前十的公司") is True
    assert _looks_like_complex_query("宁德时代2024年营业收入") is False


@pytest.mark.asyncio
async def test_planner_routes_complex_question_to_text_to_sql_without_llm():
    state = {
        "messages": [HumanMessage(content="2024年营收排名前十的公司")],
        "sub_question": "2024年营收排名前十的公司",
    }

    with patch(
        "app.agents.finance_agent.financial_query_agent.planner.node.get_router_llm",
    ) as mock_get_llm:
        result = await financial_query_planner(state)

    mock_get_llm.assert_not_called()
    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["steps"] == ["financial_query_planner"]


@pytest.mark.asyncio
async def test_planner_uses_llm_for_simple_question():
    state = {
        "messages": [HumanMessage(content="腾讯2024年营业收入")],
        "sub_question": "腾讯2024年营业收入",
    }
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=FinancialQueryPlan(
            route="predefined",
            reason="单公司单指标年度查数，命中白名单模板。",
            confidence=0.95,
        )
    )

    with patch(
        "app.agents.finance_agent.financial_query_agent.planner.node.get_router_llm",
        return_value=mock_llm,
    ):
        result = await financial_query_planner(state)

    assert result["financial_query_plan_route"] == "predefined"
    assert "白名单" in result["financial_query_plan_reason"]


@pytest.mark.asyncio
async def test_planner_empty_question_returns_database_failure():
    state = {"messages": [], "sub_question": ""}

    result = await financial_query_planner(state)

    assert result["steps"] == ["financial_query_planner_error"]
    assert "暂未在结构化财务数据库中找到相关指标" in result["messages"][0].content
