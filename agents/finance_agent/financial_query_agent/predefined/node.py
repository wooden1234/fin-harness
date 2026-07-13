"""predefined 分支入口：tool_selection → semantic → sql_builder → execution → formatter。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    database_failure_output,
    financial_query_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.predefined.execution import (
    execute_predefined_sql,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.normalizer import (
    build_predefined_query_intent,
)
from agents.finance_agent.financial_query_agent.predefined.formatter import (
    format_predefined_answer,
)
from agents.finance_agent.financial_query_agent.predefined.resolver import (
    build_resolved_query,
    build_resolved_query_from_slots,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.nodes import (
    resolve_canonical_metrics_node,
    resolve_coverage_node,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection import (
    PredefinedToolSelectionResult,
    select_predefined_tool,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

FINANCIAL_QUERY_NO_RESULT_ANSWER = "暂未在结构化财务数据库中找到相关指标，建议查阅年报 PDF 文档获取更多信息。"


def _fallback_to_text_to_sql(question: str, reason: str) -> dict:
    """predefined 无法可靠选工具时，把控制权交回复杂查询路径。"""
    return {
        "financial_query_text": question,
        "financial_query_plan_route": "text_to_sql",
        "financial_query_plan_reason": reason,
        "financial_query_template_id": None,
        "financial_query_next_action_sql": "fallback_to_text_to_sql",
        "steps": ["predefined_tool_selection_failed"],
    }


def _clarify_output(state: FinAgentState, question: str, coverage, base_updates: dict) -> dict:
    reason = coverage.clarify_reason or "当前问题存在多种可用口径，请补充查询粒度"
    return {
        **base_updates,
        **financial_query_output(
            state,
            answer=reason,
            step="predefined",
        ),
    }


async def predefined_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """assistgen 模式：tool_selection → semantic → execution → formatter。"""
    question = str(state.get("financial_query_text") or query_from_state(state)).strip()
    if not question:
        logger.error("predefined_workflow missing question")
        return database_failure_output(state, step="predefined")

    selection = await select_predefined_tool(question, config)
    if isinstance(selection, tuple):
        template_id, intent = selection
    elif isinstance(selection, PredefinedToolSelectionResult):
        if not selection.success or selection.slots is None:
            logger.warning(
                "predefined_workflow fallback to text_to_sql reason={}",
                selection.error,
            )
            return _fallback_to_text_to_sql(
                question,
                selection.error or "predefined_tool_selection_failed",
            )
        template_id = selection.template_id
        intent = build_predefined_query_intent(selection.slots)
    else:
        logger.warning("predefined_workflow unexpected tool_selection result")
        return _fallback_to_text_to_sql(question, "predefined_tool_selection_invalid_result")

    canonical_matches = await resolve_canonical_metrics_node(intent)
    coverage = await resolve_coverage_node(intent, canonical_matches, template_id)

    if coverage.status == "clarify":
        return _clarify_output(
            state,
            question,
            coverage,
            {
                "financial_query_text": question,
                "financial_query_template_id": template_id,
                "financial_query_intent": intent,
            },
        )

    if isinstance(selection, tuple):
        resolved_query = await build_resolved_query(
            template_id,
            intent,
            canonical_matches,
            coverage,
        )
    else:
        resolved_query = await build_resolved_query_from_slots(
            template_id,
            selection.slots,
            canonical_matches,
            coverage,
        )

    execution = await execute_predefined_sql(
        {
            "task": question,
            "query_name": "predefined_sql",
            "query_parameters": {"template_id": template_id},
            "intent": resolved_query.intent,
            "resolved_query": resolved_query,
            "steps": ["tool_selection", "canonical_metric_registry", "coverage_resolver"],
        },
        limit=resolved_query.intent.top_k,
    )

    base_updates = {
        "financial_query_text": question,
        "financial_query_template_id": template_id,
        "financial_query_intent": resolved_query.intent,
        "financial_query_sql": execution["statement"],
        "financial_query_sql_params": execution["parameters"],
        "financial_query_missing_fields": execution["missing_fields"],
    }
    if execution["missing_fields"] or execution["errors"]:
        logger.error(
            "predefined_workflow execution failed template_id={} missing_fields={} errors={}",
            template_id,
            execution["missing_fields"],
            execution["errors"],
        )
        return {
            **base_updates,
            **database_failure_output(state, step="predefined"),
        }

    rows = execution["rows"]
    if not rows:
        logger.info(
            "predefined_workflow no rows template_id={} company={} year={} metric={}",
            template_id,
            resolved_query.intent.company,
            resolved_query.intent.year,
            resolved_query.intent.metric,
        )
        return {
            **base_updates,
            **financial_query_output(
                state,
                answer=FINANCIAL_QUERY_NO_RESULT_ANSWER,
                step="predefined",
            ),
        }

    answer = format_predefined_answer(rows, coverage)
    logger.info("predefined_workflow rows={} template_id={}", len(rows), template_id)
    return {
        **base_updates,
        **financial_query_output(
            state,
            answer=answer,
            step="predefined",
        ),
    }


__all__ = ["predefined_workflow"]
