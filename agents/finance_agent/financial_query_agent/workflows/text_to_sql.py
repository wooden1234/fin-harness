"""text_to_sql 显式状态机 workflow。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import tracing_context

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    financial_query_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.text_to_sql import (
    correct_sql,
    execute_generated_sql,
    generate_sql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.components import (
    clarify_after_correct_node,
    clarify_after_generate_node,
    clarify_before_generate_node,
    clarify_output_node,
    correct_sql_node,
    execute_sql_node,
    execution_error_output_node,
    format_output_node,
    generate_sql_node,
    prepare_context_node,
    unsafe_output_node,
    validate_sql_node,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlNextStep,
    TextToSqlState,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

TextToSqlNode = Callable[[TextToSqlState, RunnableConfig | None], Awaitable[dict[str, Any]]]

_NODE_BY_STEP: dict[TextToSqlNextStep, TextToSqlNode] = {
    "clarify_before_generate": clarify_before_generate_node,
    "generate_sql": generate_sql_node,
    "clarify_after_generate": clarify_after_generate_node,
    "validate_sql": validate_sql_node,
    "correct_sql": correct_sql_node,
    "clarify_after_correct": clarify_after_correct_node,
    "execute_sql": execute_sql_node,
    "clarify_output": clarify_output_node,
    "unsafe_output": unsafe_output_node,
    "execution_error_output": execution_error_output_node,
    "format_output": format_output_node,
}

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
    """构建 text_to_sql 状态图，供结构测试、导出和可视化使用。"""
    builder = StateGraph(TextToSqlState)

    builder.add_node("prepare_context", prepare_context_node)
    builder.add_node("clarify_before_generate", clarify_before_generate_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("clarify_after_generate", clarify_after_generate_node)
    builder.add_node("validate_sql", validate_sql_node)
    builder.add_node("correct_sql", correct_sql_node)
    builder.add_node("clarify_after_correct", clarify_after_correct_node)
    builder.add_node("execute_sql", execute_sql_node)
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
            "execute_sql": "execute_sql",
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
        "execute_sql",
        _route_next_step,
        {
            "format_output": "format_output",
            "execution_error_output": "execution_error_output",
            END: END,
        },
    )
    builder.add_edge("clarify_output", END)
    builder.add_edge("unsafe_output", END)
    builder.add_edge("execution_error_output", END)
    builder.add_edge("format_output", END)

    return builder


async def _apply_node(
    state: TextToSqlState,
    node: TextToSqlNode,
    config: RunnableConfig = None,
) -> TextToSqlState:
    updates = await node(state, config)
    return {**state, **updates}


async def _run_text_to_sql(question: str, config: RunnableConfig = None) -> TextToSqlState:
    """按显式状态机运行 text_to_sql，所有中间态写入 TextToSqlState。"""
    current_state: TextToSqlState = {"question": question}
    current_state = await _apply_node(current_state, prepare_context_node, config)

    max_transitions = int(current_state.get("max_attempts", 3)) * 4 + 8
    for _ in range(max_transitions):
        next_step = current_state.get("next_step", "end")
        if next_step == "end":
            return current_state
        node = _NODE_BY_STEP.get(next_step)
        if node is None:
            logger.error("text_to_sql_workflow unknown next_step={}", next_step)
            current_state = {
                **current_state,
                "execution_error": "unknown_next_step",
                "next_step": "execution_error_output",
            }
            return await _apply_node(current_state, execution_error_output_node, config)
        current_state = await _apply_node(current_state, node, config)

    logger.error("text_to_sql_workflow exceeded transition limit")
    current_state = {
        **current_state,
        "execution_error": "transition_limit_exceeded",
        "next_step": "execution_error_output",
    }
    return await _apply_node(current_state, execution_error_output_node, config)


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


async def text_to_sql_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """运行 text_to_sql 状态机，并将局部状态适配为 FinAgentState 更新。"""
    question = str(state.get("financial_query_text") or query_from_state(state)).strip()
    invoke_config: RunnableConfig = {**(config or {}), "callbacks": []}
    with tracing_context(enabled=False):
        result = await _run_text_to_sql(question, invoke_config)

    answer = str(result.get("answer", ""))
    if not answer:
        logger.error("text_to_sql_workflow ended without answer final_status={}", result.get("final_status"))
        result = {
            **result,
            "answer": "当前结构化查询未能生成有效答案，请补充更具体的查询条件后重试。",
            "final_status": "execution_error",
        }
        answer = result["answer"]

    return {
        **_base_updates(question, result),
        **financial_query_output(state, answer=answer, step="text_to_sql"),
        "financial_query_next_action_sql": _next_action(result),
    }


__all__ = [
    "build_text_to_sql_workflow_graph",
    "correct_sql",
    "execute_generated_sql",
    "generate_sql",
    "text_to_sql_workflow",
]
