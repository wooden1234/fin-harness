"""text_to_sql 显式状态机 workflow。

校验链路：
  validate_sql（规则门）→ db_verify（真库）→ validate_result（规则 + 可选 LLM 质检）→ format_output
规则失败 → correct_sql；库失败可重试 → correct_sql；结果可疑 → correct_sql 或 execution_error。
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import tracing_context

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    financial_query_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.text_to_sql.components import (
    clarify_after_correct_node,
    clarify_after_generate_node,
    clarify_before_generate_node,
    clarify_output_node,
    correct_sql_node,
    db_verify_node,
    execution_error_output_node,
    format_output_node,
    generate_sql_node,
    prepare_context_node,
    unsafe_output_node,
    validate_result_node,
    validate_sql_node,
)
from agents.finance_agent.financial_query_agent.text_to_sql.components.nodes import (
    DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS,
    FINANCIAL_QUERY_TEXT_TO_SQL_FALLBACK_ANSWER,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlNextStep,
    TextToSqlState,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

_TERMINAL_NEXT_ACTION_BY_STATUS = {
    "clarify": "clarify",
    "unsafe": "end",
    "execution_error": "end",
    "success": "execute",
}


def _route_next_step(state: TextToSqlState) -> TextToSqlNextStep | str:
    next_step = state.get("next_step", "end")
    if next_step == "end":
        return END
    return next_step


def build_text_to_sql_workflow_graph() -> StateGraph:
    """构建 text_to_sql 状态图。"""
    builder = StateGraph(TextToSqlState)

    builder.add_node("prepare_context", prepare_context_node)
    builder.add_node("clarify_before_generate", clarify_before_generate_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("clarify_after_generate", clarify_after_generate_node)
    builder.add_node("validate_sql", validate_sql_node)
    builder.add_node("correct_sql", correct_sql_node)
    builder.add_node("clarify_after_correct", clarify_after_correct_node)
    builder.add_node("db_verify", db_verify_node)
    builder.add_node("validate_result", validate_result_node)
    builder.add_node("clarify_output", clarify_output_node)
    builder.add_node("unsafe_output", unsafe_output_node)
    builder.add_node("execution_error_output", execution_error_output_node)
    builder.add_node("format_output", format_output_node)

    builder.add_edge(START, "prepare_context")
    builder.add_conditional_edges(
        "prepare_context",
        _route_next_step,
        {
            "clarify_before_generate": "clarify_before_generate",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "clarify_before_generate",
        _route_next_step,
        {
            "generate_sql": "generate_sql",
            "clarify_output": "clarify_output",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "generate_sql",
        _route_next_step,
        {
            "clarify_after_generate": "clarify_after_generate",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "clarify_after_generate",
        _route_next_step,
        {
            "validate_sql": "validate_sql",
            "clarify_output": "clarify_output",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "validate_sql",
        _route_next_step,
        {
            "db_verify": "db_verify",
            "correct_sql": "correct_sql",
            "unsafe_output": "unsafe_output",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "correct_sql",
        _route_next_step,
        {
            "clarify_after_correct": "clarify_after_correct",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "clarify_after_correct",
        _route_next_step,
        {
            "validate_sql": "validate_sql",
            "clarify_output": "clarify_output",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "db_verify",
        _route_next_step,
        {
            "validate_result": "validate_result",
            "correct_sql": "correct_sql",
            "execution_error_output": "execution_error_output",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "validate_result",
        _route_next_step,
        {
            "format_output": "format_output",
            "correct_sql": "correct_sql",
            "execution_error_output": "execution_error_output",
            END: END,
        },
    )
    builder.add_edge("clarify_output", END)
    builder.add_edge("unsafe_output", END)
    builder.add_edge("execution_error_output", END)
    builder.add_edge("format_output", END)

    return builder


_COMPILED_TEXT_TO_SQL_GRAPH = None


def _get_compiled_text_to_sql_graph():
    global _COMPILED_TEXT_TO_SQL_GRAPH
    if _COMPILED_TEXT_TO_SQL_GRAPH is None:
        _COMPILED_TEXT_TO_SQL_GRAPH = build_text_to_sql_workflow_graph().compile()
    return _COMPILED_TEXT_TO_SQL_GRAPH


def _recursion_limit(max_attempts: int = DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS) -> int:
    return max_attempts * 6 + 12


def _execution_error_result(
    *,
    question: str,
    execution_error: str = "transition_limit_exceeded",
) -> TextToSqlState:
    return {
        "question": question,
        "execution_error": execution_error,
        "answer": FINANCIAL_QUERY_TEXT_TO_SQL_FALLBACK_ANSWER,
        "final_status": "execution_error",
    }


def _base_updates(question: str, result: TextToSqlState) -> dict[str, Any]:
    return {
        "financial_query_text": question,
        "financial_query_schema_prompt": str(result.get("schema_prompt", "")),
        "financial_query_fewshot_examples": str(result.get("fewshot_examples", "")),
        "financial_query_sql_attempts": int(result.get("attempts", 0)),
        "financial_query_sql": str(result.get("sql", "")),
        "financial_query_sql_params": dict(result.get("sql_params", {})),
        "financial_query_validated_sql": str(result.get("validated_sql", "")),
        "financial_query_validation_error": str(result.get("validation_error", "")),
        "financial_query_validation_error_type": str(result.get("validation_error_type", "")),
        "financial_query_validation_errors": list(result.get("validation_errors", [])),
        "financial_query_validation_error_types": list(result.get("validation_error_types", [])),
        "financial_query_missing_fields": list(result.get("missing_fields", [])),
        "financial_query_plan_reason": str(result.get("route_reason", "")),
    }


def _next_action(result: TextToSqlState) -> str:
    final_status = str(result.get("final_status", "execution_error"))
    return _TERMINAL_NEXT_ACTION_BY_STATUS.get(final_status, "end")


def _coverage_for_final_status(result: TextToSqlState) -> str:
    status = str(result.get("final_status") or "execution_error")
    if status == "success":
        return "covered"
    if status == "clarify":
        return "clarify"
    return "uncovered"


async def text_to_sql_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """运行 text_to_sql 状态机，并将局部状态适配为 FinAgentState 更新。"""
    question = str(state.get("financial_query_text") or query_from_state(state)).strip()
    invoke_config: RunnableConfig = {
        **(config or {}),
        "callbacks": [],
        "recursion_limit": _recursion_limit(),
    }
    try:
        with tracing_context(enabled=False):
            result = await _get_compiled_text_to_sql_graph().ainvoke(
                {"question": question},
                config=invoke_config,
            )
    except GraphRecursionError:
        logger.error(
            "text_to_sql_workflow exceeded recursion_limit={}",
            _recursion_limit(),
        )
        result = _execution_error_result(question=question)

    answer = str(result.get("answer", ""))
    final_status = str(result.get("final_status") or "execution_error")
    coverage = _coverage_for_final_status(result)
    if not answer:
        logger.error("text_to_sql_workflow ended without answer final_status={}", final_status)
        result = {
            **result,
            "answer": "当前结构化查询未能生成有效答案，请补充更具体的查询条件后重试。",
            "final_status": "execution_error",
        }
        answer = result["answer"]
        coverage = "uncovered"

    fq_output = financial_query_output(
        state,
        answer=answer,
        step="text_to_sql",
        coverage=coverage,
        fallback_reason="financial_query_text_to_sql_failed" if coverage == "uncovered" else "",
    )
    return {
        **_base_updates(question, result),
        **fq_output,
        "financial_query_next_action_sql": _next_action(result),
    }


__all__ = [
    "build_text_to_sql_workflow_graph",
    "text_to_sql_workflow",
]
