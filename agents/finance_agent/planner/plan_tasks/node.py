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
    original_query = latest_user_query(list(state.get("messages") or []))
    if not original_query:
        return {
            **empty_plan(step="plan_tasks", reason="empty_query"),
            "planner_query": "",
            "planner_raw_tasks": [],
            "planner_validation_issues": ["empty_query"],
            "planner_needs_repair": False,
            "planner_repair_attempted": False,
            "planner_error_reason": "empty_query",
        }

    rewritten_query = str(state.get("rewritten_query") or "").strip()
    rewrite_status = str(state.get("rewrite_status") or "").strip()
    conversation_summary = str(state.get("conversation_summary") or "")

    # 改写节点明确判定上下文不足时，不再让 Planner 对缺失字段猜测。
    # 空计划会沿 dispatch_workers 的现有澄清兜底链返回用户可读追问。
    if rewrite_status == "uncertain":
        logger.info("planner skipped because query rewrite is uncertain")
        return {
            **empty_plan(step="plan_tasks", reason="rewrite_uncertain"),
            "planner_query": original_query,
            "planner_raw_tasks": [],
            "planner_validation_issues": ["rewrite_uncertain"],
            "planner_needs_repair": False,
            "planner_repair_attempted": False,
            "planner_error_reason": "rewrite_uncertain",
        }

    prompt_parts = [
        f"此前对话摘要：\n{conversation_summary or '无'}",
        f"当前用户问题（原文）：\n{original_query}",
    ]
    if rewritten_query and rewritten_query != original_query:
        prompt_parts.append(f"改写后的完整问题：\n{rewritten_query}")
    if rewrite_status == "uncertain":
        prompt_parts.append(
            "改写状态：uncertain（上下文不足，禁止根据猜测补全；无法形成明确子任务时返回空 tasks）"
        )
    human_prompt = "\n\n".join(prompt_parts)

    logger.info(
        "planner original={} rewritten={} summary_chars={}",
        original_query[:80],
        (rewritten_query or "-")[:80],
        len(conversation_summary),
    )

    try:
        output = await plan_with_retry(human_prompt, config)
    except Exception as exc:
        if is_transient_api_error(exc):
            logger.exception("planner api failed after retry")
            return {
                **empty_plan(step="plan_tasks", reason="api_error"),
                "planner_query": original_query,
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
                "planner_query": original_query,
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
            "planner_query": original_query,
            "planner_raw_tasks": [],
            "planner_validation_issues": ["unexpected_error"],
            "planner_needs_repair": False,
            "planner_repair_attempted": False,
            "planner_error_reason": "unexpected_error",
        }

    return {
        **begin_turn_workspace(),
        "planner_query": original_query,
        "planner_raw_tasks": list(output.tasks),
        "planner_validation_issues": [],
        "planner_needs_repair": False,
        "planner_repair_attempted": False,
        "planner_error_reason": "",
        "steps": ["plan_tasks"],
    }


__all__ = ["plan_tasks_node"]
