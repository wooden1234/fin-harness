"""predefined 白名单查询工作流编排。"""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import tracing_context
from typing_extensions import TypedDict

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    database_failure_output,
    financial_query_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.predefined.execution import (
    build_predefined_sql_query,
    execute_predefined_sql,
    execute_predefined_sql_query,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.normalizer import (
    build_predefined_query_intent,
)
from agents.finance_agent.financial_query_agent.predefined.formatter import (
    format_predefined_answer,
)
from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.resolver import (
    build_resolved_query,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricMatch,
    CoverageResolution,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.nodes import (
    resolve_canonical_metrics_node,
    resolve_coverage_node,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection import (
    PredefinedToolSelectionResult,
    select_predefined_tool,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.registry import (
    ResolvedPredefinedQuery,
)
from agents.finance_agent.financial_query_agent.services.fact_service import (
    FinancialFactService,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

FINANCIAL_QUERY_NO_RESULT_ANSWER = "暂未在结构化财务数据库中找到相关指标，建议查阅年报 PDF 文档获取更多信息。"

PredefinedRoute = Literal["select_tool", "semantic", "resolve", "execute", "format", "__end__"]


class PredefinedGraphState(TypedDict):
    """predefined 图内部状态，避免临时态泄漏到父图。"""

    source_state: FinAgentState
    question: str
    output: NotRequired[dict[str, Any]]
    selection: NotRequired[PredefinedToolSelectionResult]
    template_id: NotRequired[str]
    intent: NotRequired[FinancialQueryIntent]
    canonical_matches: NotRequired[list[CanonicalMetricMatch]]
    coverage: NotRequired[CoverageResolution]
    resolved_query: NotRequired[ResolvedPredefinedQuery]
    execution: NotRequired[dict[str, Any]]
    rows: NotRequired[list[FinancialSqlResultRow]]


def _fallback_to_text_to_sql(
    question: str,
    reason: str,
    *,
    step: str = "predefined_tool_selection_failed",
) -> dict[str, Any]:
    """predefined 无法可靠执行时，把控制权交回复杂查询路径。"""
    return {
        "financial_query_text": question,
        "financial_query_plan_route": "text_to_sql",
        "financial_query_plan_reason": reason,
        "financial_query_template_id": None,
        "financial_query_next_action_sql": "fallback_to_text_to_sql",
        "steps": [step],
    }


def _clarify_output(
    state: FinAgentState,
    question: str,
    coverage: CoverageResolution,
    base_updates: dict[str, Any],
) -> dict[str, Any]:
    reason = (
        coverage.clarify_reason
        or "当前问题存在多种可用口径，请补充查询粒度"
    )
    return {
        **base_updates,
        **financial_query_output(
            state,
            answer=reason,
            step="predefined",
            coverage="clarify",
        ),
    }


def _base_execution_updates(
    question: str,
    template_id: str,
    intent: FinancialQueryIntent,
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "financial_query_text": question,
        "financial_query_template_id": template_id,
        "financial_query_intent": intent,
        "financial_query_sql": execution["statement"],
        "financial_query_sql_params": execution["parameters"],
        "financial_query_missing_fields": execution["missing_fields"],
    }


def _route_after_init(state: PredefinedGraphState) -> PredefinedRoute:
    if state.get("output"):
        return END
    return "select_tool"


def _route_after_tool_selection(state: PredefinedGraphState) -> PredefinedRoute:
    if state.get("output"):
        return END
    return "semantic"


def _route_after_semantic(state: PredefinedGraphState) -> PredefinedRoute:
    if state.get("output"):
        return END
    return "resolve"


def _route_after_execute(state: PredefinedGraphState) -> PredefinedRoute:
    if state.get("output"):
        return END
    return "format"


async def _init_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    source_state = state["source_state"]
    question = str(
        source_state.get("financial_query_text") or query_from_state(source_state)
    ).strip()
    if not question:
        logger.error("predefined_workflow missing question")
        return {
            "question": question,
            "output": database_failure_output(source_state, step="predefined"),
        }
    return {"question": question}


async def _select_tool_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    question = state["question"]
    selection = await select_predefined_tool(question, config)

    if not selection.success or selection.slots is None:
        logger.warning(
            "predefined_workflow fallback to text_to_sql reason={}",
            selection.error,
        )
        return {
            "selection": selection,
            "output": _fallback_to_text_to_sql(
                question,
                selection.error or "predefined_tool_selection_failed",
            ),
        }

    return {
        "selection": selection,
        "template_id": selection.template_id,
        "intent": build_predefined_query_intent(selection.slots),
    }


async def _semantic_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    question = state["question"]
    source_state = state["source_state"]
    template_id = state["template_id"]
    intent = state["intent"]
    canonical_matches = await resolve_canonical_metrics_node(intent)
    coverage = await resolve_coverage_node(intent, canonical_matches, template_id)
    if coverage.status == "clarify":
        return {
            "canonical_matches": canonical_matches,
            "coverage": coverage,
            "output": _clarify_output(
                source_state,
                question,
                coverage,
                {
                    "financial_query_text": question,
                    "financial_query_template_id": template_id,
                    "financial_query_intent": intent,
                },
            ),
        }
    if coverage.status == "unavailable":
        reason_code = coverage.reason_code or "ANNUAL_DATA_NOT_FOUND"
        logger.warning(
            "predefined_workflow coverage unavailable → text_to_sql reason_code={}",
            reason_code,
        )
        return {
            "canonical_matches": canonical_matches,
            "coverage": coverage,
            "output": {
                "financial_query_text": question,
                "financial_query_template_id": template_id,
                "financial_query_intent": intent,
                **_fallback_to_text_to_sql(
                    question,
                    reason_code,
                    step=reason_code,
                ),
            },
        }
    return {
        "canonical_matches": canonical_matches,
        "coverage": coverage,
    }


async def _resolve_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    template_id = state["template_id"]
    canonical_matches = state["canonical_matches"]
    coverage = state["coverage"]
    resolved_query = await build_resolved_query(
        template_id,
        state["intent"],
        canonical_matches,
        coverage,
    )
    return {"resolved_query": resolved_query}


async def _execute_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    question = state["question"]
    source_state = state["source_state"]
    template_id = state["template_id"]
    resolved_query = state["resolved_query"]
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

    base_updates = _base_execution_updates(
        question,
        template_id,
        resolved_query.intent,
        execution,
    )
    if execution["missing_fields"] or execution["errors"]:
        logger.error(
            "predefined_workflow execution failed template_id={} missing_fields={} errors={}",
            template_id,
            execution["missing_fields"],
            execution["errors"],
        )
        return {
            "execution": execution,
            "output": {
                **base_updates,
                **database_failure_output(source_state, step="predefined"),
            },
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
            "execution": execution,
            "rows": rows,
            "output": {
                **base_updates,
                **financial_query_output(
                    source_state,
                    answer=FINANCIAL_QUERY_NO_RESULT_ANSWER,
                    step="predefined",
                    coverage="uncovered",
                    fallback_reason="financial_query_no_rows",
                ),
            },
        }

    return {
        "execution": execution,
        "rows": rows,
    }


async def _format_node(
    state: PredefinedGraphState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    question = state["question"]
    source_state = state["source_state"]
    template_id = state["template_id"]
    resolved_query = state["resolved_query"]
    execution = state["execution"]
    coverage = state["coverage"]
    answer = format_predefined_answer(state["rows"], coverage)
    logger.info("predefined_workflow rows={} template_id={}", len(state["rows"]), template_id)
    return {
        "output": {
            **_base_execution_updates(
                question,
                template_id,
                resolved_query.intent,
                execution,
            ),
            **financial_query_output(
                source_state,
                answer=answer,
                step="predefined",
            ),
        }
    }


def build_predefined_workflow_graph() -> StateGraph:
    """构建 predefined 白名单查询图。"""
    builder = StateGraph(PredefinedGraphState)

    builder.add_node("init", _init_node)
    builder.add_node("select_tool", _select_tool_node)
    builder.add_node("semantic", _semantic_node)
    builder.add_node("resolve", _resolve_node)
    builder.add_node("execute", _execute_node)
    builder.add_node("format", _format_node)

    builder.add_edge(START, "init")
    builder.add_conditional_edges(
        "init",
        _route_after_init,
        {
            "select_tool": "select_tool",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "select_tool",
        _route_after_tool_selection,
        {
            "semantic": "semantic",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "semantic",
        _route_after_semantic,
        {
            "resolve": "resolve",
            END: END,
        },
    )
    builder.add_edge("resolve", "execute")
    builder.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "format": "format",
            END: END,
        },
    )
    builder.add_edge("format", END)

    return builder


_COMPILED_PREDEFINED_GRAPH = None


def _get_compiled_predefined_graph():
    global _COMPILED_PREDEFINED_GRAPH
    if _COMPILED_PREDEFINED_GRAPH is None:
        _COMPILED_PREDEFINED_GRAPH = build_predefined_workflow_graph().compile()
    return _COMPILED_PREDEFINED_GRAPH


async def predefined_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """运行 predefined 白名单查询图，并保持旧 workflow 的对外返回契约。"""
    invoke_config: RunnableConfig = {**(config or {}), "callbacks": []}
    with tracing_context(enabled=False):
        graph_state = await _get_compiled_predefined_graph().ainvoke(
            {"source_state": state, "question": ""},
            config=invoke_config,
        )
    output = graph_state.get("output")
    if isinstance(output, dict):
        return output

    logger.error("predefined_workflow graph ended without output")
    return database_failure_output(state, step="predefined")


__all__ = [
    "FinancialFactService",
    "build_predefined_sql_query",
    "build_predefined_workflow_graph",
    "execute_predefined_sql_query",
    "predefined_workflow",
]
