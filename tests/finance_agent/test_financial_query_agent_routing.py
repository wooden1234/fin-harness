"""financial_query_agent 顶层路由与 predefined→text_to_sql 降级契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents.finance_agent.financial_query_agent import (
    _run_financial_query_agent,
)


def _base_state(question: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "sub_question": question,
        "sub_task_id": "task-1",
    }


@pytest.mark.asyncio
async def test_agent_routes_to_predefined_when_planner_says_so():
    state = _base_state("腾讯2024年营业收入")
    planner_updates = {
        "financial_query_text": state["sub_question"],
        "financial_query_plan_route": "predefined",
        "financial_query_plan_reason": "命中白名单",
        "steps": ["financial_query_planner"],
    }
    predefined_updates = {
        "messages": [HumanMessage(content="腾讯 2024 年营业收入为 660,257 百万元。")],
        "task_results": [{"sub_task_id": "task-1", "type": "financial_query"}],
        "steps": ["predefined"],
    }

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.planner.financial_query_planner",
            new=AsyncMock(return_value=planner_updates),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined_workflow",
            new=AsyncMock(return_value=predefined_updates),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.text_to_sql_workflow",
            new=AsyncMock(),
        ) as mock_text_to_sql,
    ):
        result = await _run_financial_query_agent(state)

    mock_text_to_sql.assert_not_called()
    assert result["financial_query_plan_route"] == "predefined"
    assert result["steps"] == ["financial_query_planner", "predefined"]


@pytest.mark.asyncio
async def test_agent_falls_back_to_text_to_sql_when_predefined_fails():
    state = _base_state("腾讯2024年营业收入")
    planner_updates = {
        "financial_query_text": state["sub_question"],
        "financial_query_plan_route": "predefined",
        "financial_query_plan_reason": "命中白名单",
        "steps": ["financial_query_planner"],
    }
    predefined_updates = {
        "financial_query_plan_route": "text_to_sql",
        "financial_query_next_action_sql": "fallback_to_text_to_sql",
        "steps": ["predefined_tool_selection_failed"],
    }
    text_to_sql_updates = {
        "messages": [HumanMessage(content="text_to_sql 答案")],
        "task_results": [{"sub_task_id": "task-1", "type": "financial_query"}],
        "steps": ["text_to_sql"],
    }

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.planner.financial_query_planner",
            new=AsyncMock(return_value=planner_updates),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined_workflow",
            new=AsyncMock(return_value=predefined_updates),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.text_to_sql_workflow",
            new=AsyncMock(return_value=text_to_sql_updates),
        ) as mock_text_to_sql,
    ):
        result = await _run_financial_query_agent(state)

    mock_text_to_sql.assert_awaited_once()
    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["steps"] == [
        "financial_query_planner",
        "predefined_tool_selection_failed",
        "text_to_sql",
    ]


@pytest.mark.asyncio
async def test_agent_routes_directly_to_text_to_sql():
    state = _base_state("2024年营收排名前十的公司")
    planner_updates = {
        "financial_query_text": state["sub_question"],
        "financial_query_plan_route": "text_to_sql",
        "financial_query_plan_reason": "复杂查询",
        "steps": ["financial_query_planner"],
    }
    text_to_sql_updates = {
        "messages": [HumanMessage(content="text_to_sql 答案")],
        "task_results": [{"sub_task_id": "task-1", "type": "financial_query"}],
        "steps": ["text_to_sql"],
    }

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.planner.financial_query_planner",
            new=AsyncMock(return_value=planner_updates),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined_workflow",
            new=AsyncMock(),
        ) as mock_predefined,
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.text_to_sql_workflow",
            new=AsyncMock(return_value=text_to_sql_updates),
        ),
    ):
        result = await _run_financial_query_agent(state)

    mock_predefined.assert_not_called()
    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["steps"] == ["financial_query_planner", "text_to_sql"]
