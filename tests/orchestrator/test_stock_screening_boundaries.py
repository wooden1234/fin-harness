"""选股 Planner 的用户约束边界测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agents.orchestrator.contracts import (
    AgentResult,
    MarketFilter,
    MarketQueryPlan,
)
from agents.stock_screening_agent.workflow import (
    ScreeningPlanDraft,
    build_stock_screening_workflow,
    run_stock_screening_workflow,
)


def test_stock_screening_workflow_has_no_automatic_retry_branch() -> None:
    graph = build_stock_screening_workflow().get_graph()

    assert {"parse", "validate", "execute", "finalize"} <= set(graph.nodes)
    assert "evaluate" not in graph.nodes
    assert "retry" not in graph.draw_mermaid()


def test_screening_draft_cannot_request_adjustment() -> None:
    properties = ScreeningPlanDraft.model_json_schema()["properties"]

    assert "adjustment" not in properties


@pytest.mark.asyncio
async def test_empty_result_preserves_explicit_numeric_threshold(
    monkeypatch,
) -> None:
    planner = AsyncMock(
        return_value=ScreeningPlanDraft(
            plan=MarketQueryPlan(
                universe="A股",
                filters=[
                    MarketFilter(
                        field="industry",
                        operator="contains",
                        value="新能源",
                    ),
                    MarketFilter(
                        field="pe_ttm",
                        operator="lt",
                        value=20,
                    ),
                ],
            )
        )
    )
    calls: list[tuple[str, str]] = []

    async def fake_tool(*, query, task_input, **kwargs):
        del kwargs
        calls.append((query, task_input["call_type"]))
        return AgentResult(
            task_id="stock-screening",
            agent_id="stock_screening_agent",
            status="uncovered",
            answer="未找到候选股票",
            error_code="iwencai.screen_empty_result",
        )

    monkeypatch.setattr(
        "agents.stock_screening_agent.workflow.parse_screening_plan",
        planner,
    )
    monkeypatch.setattr(
        "agents.stock_screening_agent.workflow.run_iwencai_source_tool",
        fake_tool,
    )

    result = await run_stock_screening_workflow(
        {},
        query="筛选新能源行业、市盈率低于20的股票",
    )

    assert result.status == "uncovered"
    assert calls == [
        ("A股，行业包含新能源，市盈率TTM小于20，取前20只", "normal")
    ]
    assert planner.await_count == 1
    assert result.metadata["retry_count"] == 0
