"""predefined 工作流分支契约测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agents.finance_agent.financial_query_agent.predefined.extraction.models import (
    PredefinedSlotExtraction,
)
from app.agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from app.agents.finance_agent.financial_query_agent.predefined.node import (
    predefined_workflow,
)
from app.agents.finance_agent.financial_query_agent.workflows.predefined import (
    predefined_workflow as graph_predefined_workflow,
)
from app.agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CoverageResolution,
)
from app.agents.finance_agent.financial_query_agent.predefined.tool_selection.node import (
    PredefinedToolSelectionResult,
)
from app.agents.finance_agent.financial_query_agent.predefined.whitelist.registry import (
    ResolvedPredefinedQuery,
)
from app.agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)


def _base_state(question: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "sub_question": question,
        "sub_task_id": "task-1",
        "financial_query_text": question,
    }


@pytest.mark.asyncio
async def test_predefined_tool_selection_failure_fallbacks_to_text_to_sql():
    state = _base_state("腾讯2024年营业收入")
    failed_selection = PredefinedToolSelectionResult(
        success=False,
        template_id="",
        slots=None,
        error="predefined_tool_selection_failed",
    )

    with patch(
        "app.agents.finance_agent.financial_query_agent.predefined.node.select_predefined_tool",
        new=AsyncMock(return_value=failed_selection),
    ):
        result = await predefined_workflow(state)

    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["financial_query_next_action_sql"] == "fallback_to_text_to_sql"
    assert result["steps"] == ["predefined_tool_selection_failed"]


@pytest.mark.asyncio
async def test_predefined_coverage_clarify_returns_clarify_answer():
    state = _base_state("腾讯2024年营业收入")
    slots = PredefinedSlotExtraction(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    selection = PredefinedToolSelectionResult(
        success=True,
        template_id="exact_metric_lookup",
        slots=slots,
    )
    coverage = CoverageResolution(
        status="clarify",
        clarify_reason="当前问题存在多种可用口径，请补充查询粒度",
    )

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.select_predefined_tool",
            new=AsyncMock(return_value=selection),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.resolve_canonical_metrics_node",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.resolve_coverage_node",
            new=AsyncMock(return_value=coverage),
        ),
    ):
        result = await predefined_workflow(state)

    assert coverage.clarify_reason in result["messages"][0].content
    assert result["steps"] == ["predefined"]
    assert result["task_results"][0]["type"] == "financial_query"


@pytest.mark.asyncio
async def test_predefined_happy_path_returns_formatted_answer():
    state = _base_state("腾讯2024年营业收入")
    slots = PredefinedSlotExtraction(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    selection = PredefinedToolSelectionResult(
        success=True,
        template_id="exact_metric_lookup",
        slots=slots,
    )
    intent = FinancialQueryIntent(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    resolved_query = ResolvedPredefinedQuery(
        template_id="exact_metric_lookup",
        intent=intent,
        company_ids=[3],
    )
    coverage = CoverageResolution(status="ok")
    rows = [
        FinancialSqlResultRow(
            company_name="腾讯",
            fiscal_year=2024,
            metric_name="营业收入",
            raw_value="660,257",
            unit="百万元",
        )
    ]

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.select_predefined_tool",
            new=AsyncMock(return_value=selection),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.resolve_canonical_metrics_node",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.resolve_coverage_node",
            new=AsyncMock(return_value=coverage),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.build_resolved_query_from_slots",
            new=AsyncMock(return_value=resolved_query),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.execute_predefined_sql",
            new=AsyncMock(
                return_value={
                    "statement": "SELECT ...",
                    "parameters": {"company_id": 3},
                    "missing_fields": [],
                    "errors": [],
                    "rows": rows,
                }
            ),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.predefined.node.format_predefined_answer",
            return_value="腾讯 2024 年营业收入为 660,257 百万元。",
        ),
    ):
        result = await predefined_workflow(state)

    assert "660,257" in result["messages"][0].content
    assert result["financial_query_template_id"] == "exact_metric_lookup"
    assert result["steps"] == ["predefined"]


@pytest.mark.asyncio
async def test_graph_predefined_tool_selection_failure_fallbacks_to_text_to_sql():
    state = _base_state("腾讯2024年营业收入")
    failed_selection = PredefinedToolSelectionResult(
        success=False,
        template_id="",
        slots=None,
        error="predefined_tool_selection_failed",
    )

    with patch(
        "app.agents.finance_agent.financial_query_agent.workflows.predefined.select_predefined_tool",
        new=AsyncMock(return_value=failed_selection),
    ):
        result = await graph_predefined_workflow(state)

    assert result["financial_query_plan_route"] == "text_to_sql"
    assert result["financial_query_next_action_sql"] == "fallback_to_text_to_sql"
    assert result["steps"] == ["predefined_tool_selection_failed"]


@pytest.mark.asyncio
async def test_graph_predefined_coverage_clarify_returns_clarify_answer():
    state = _base_state("腾讯2024年营业收入")
    slots = PredefinedSlotExtraction(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    selection = PredefinedToolSelectionResult(
        success=True,
        template_id="exact_metric_lookup",
        slots=slots,
    )
    coverage = CoverageResolution(
        status="clarify",
        clarify_reason="当前问题存在多种可用口径，请补充查询粒度",
    )

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.select_predefined_tool",
            new=AsyncMock(return_value=selection),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.resolve_canonical_metrics_node",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.resolve_coverage_node",
            new=AsyncMock(return_value=coverage),
        ),
    ):
        result = await graph_predefined_workflow(state)

    assert coverage.clarify_reason in result["messages"][0].content
    assert result["financial_query_template_id"] == "exact_metric_lookup"
    assert result["steps"] == ["predefined"]


@pytest.mark.asyncio
async def test_graph_predefined_happy_path_returns_formatted_answer():
    state = _base_state("腾讯2024年营业收入")
    slots = PredefinedSlotExtraction(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    selection = PredefinedToolSelectionResult(
        success=True,
        template_id="exact_metric_lookup",
        slots=slots,
    )
    intent = FinancialQueryIntent(
        companies=["腾讯"],
        years=[2024],
        metrics=["营业收入"],
        operation="lookup",
    )
    resolved_query = ResolvedPredefinedQuery(
        template_id="exact_metric_lookup",
        intent=intent,
        company_ids=[3],
    )
    coverage = CoverageResolution(status="ok")
    rows = [
        FinancialSqlResultRow(
            company_name="腾讯",
            fiscal_year=2024,
            metric_name="营业收入",
            raw_value="660,257",
            unit="百万元",
        )
    ]

    with (
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.select_predefined_tool",
            new=AsyncMock(return_value=selection),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.resolve_canonical_metrics_node",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.resolve_coverage_node",
            new=AsyncMock(return_value=coverage),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.build_resolved_query_from_slots",
            new=AsyncMock(return_value=resolved_query),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.execute_predefined_sql",
            new=AsyncMock(
                return_value={
                    "statement": "SELECT ...",
                    "parameters": {"company_id": 3},
                    "missing_fields": [],
                    "errors": [],
                    "rows": rows,
                }
            ),
        ),
        patch(
            "app.agents.finance_agent.financial_query_agent.workflows.predefined.format_predefined_answer",
            return_value="腾讯 2024 年营业收入为 660,257 百万元。",
        ),
    ):
        result = await graph_predefined_workflow(state)

    assert "660,257" in result["messages"][0].content
    assert result["financial_query_template_id"] == "exact_metric_lookup"
    assert result["steps"] == ["predefined"]
