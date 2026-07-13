"""plan_tasks 节点：只负责 LLM 拆分原始子任务。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.finance_agent.planner.common import (
    begin_turn_workspace,
    empty_plan,
    is_schema_error,
    is_transient_api_error,
    latest_user_query,
    logger,
    plan_with_retry,
)


async def plan_tasks_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """规划原始子任务，只负责 LLM 拆分与 API/Schema 失败分级。"""
    query = latest_user_query(list(state.get("messages") or []))
    if not query:
        return {
            **empty_plan(step="plan_tasks", reason="empty_query"),
            "planner_query": "",
            "planner_raw_tasks": [],
            "planner_validation_issues": ["empty_query"],
            "planner_needs_repair": False,
            "planner_repair_attempted": False,
            "planner_error_reason": "empty_query",
        }

    logger.info("planner query={}", query[:120])

    try:
        output = await plan_with_retry(query, config)
    except Exception as exc:
        if is_transient_api_error(exc):
            logger.exception("planner api failed after retry")
            return {
                **empty_plan(step="plan_tasks", reason="api_error"),
                "planner_query": query,
                "planner_raw_tasks": [],
                "planner_validation_issues": ["api_error"],
                "planner_needs_repair": False,
                "planner_repair_attempted": False,
                "planner_error_reason": "api_error",
            }
        if is_schema_error(exc):
            logger.warning(
                "planner schema/parse error, deferring to repair node: {}",
                type(exc).__name__,
            )
            return {
                **begin_turn_workspace(),
                "planner_query": query,
                "planner_raw_tasks": [],
                "planner_validation_issues": [f"schema_error:{type(exc).__name__}"],
                "planner_needs_repair": True,
                "planner_repair_attempted": False,
                "planner_error_reason": "schema_error",
                "sub_tasks": [],
                "steps": ["plan_tasks:schema_error"],
            }
        logger.exception("planner failed with unexpected error")
        return {
            **empty_plan(step="plan_tasks", reason="unexpected_error"),
            "planner_query": query,
            "planner_raw_tasks": [],
            "planner_validation_issues": ["unexpected_error"],
            "planner_needs_repair": False,
            "planner_repair_attempted": False,
            "planner_error_reason": "unexpected_error",
        }

    return {
        **begin_turn_workspace(),
        "planner_query": query,
        "planner_raw_tasks": list(output.tasks),
        "planner_validation_issues": [],
        "planner_needs_repair": False,
        "planner_repair_attempted": False,
        "planner_error_reason": "",
        "steps": ["plan_tasks"],
    }


__all__ = ["plan_tasks_node"]
