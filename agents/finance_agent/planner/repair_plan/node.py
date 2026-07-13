"""repair_plan 节点：修复非法 planner 输出。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.finance_agent.planner.common import (
    assign_task_ids,
    logger,
    repair_plan,
)
from agents.finance_agent.planner.validate import validate_and_normalize_tasks


async def repair_plan_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """修复非法 planner 输出，并在节点内完成二次校验。"""
    query = str(state.get("planner_query") or "")
    raw_tasks = list(state.get("planner_raw_tasks") or [])
    issues = list(state.get("planner_validation_issues") or [])

    try:
        repaired = await repair_plan(query, raw_tasks, issues, config)
        validation = validate_and_normalize_tasks(repaired.tasks)
    except Exception:
        logger.exception("planner repair invoke failed")
        return {
            "sub_tasks": [],
            "planner_raw_tasks": raw_tasks,
            "planner_needs_repair": False,
            "planner_repair_attempted": True,
            "planner_error_reason": "repair_failed",
            "steps": ["repair_plan:failed"],
        }

    repaired_issues = issues + validation.issues
    tasks = assign_task_ids(validation.tasks)
    if validation.needs_repair or not tasks:
        reason = "invalid_after_repair" if validation.needs_repair else "repair_empty"
        logger.warning("planner repair unresolved reason={} issues={}", reason, repaired_issues)
        return {
            "sub_tasks": [],
            "planner_raw_tasks": list(repaired.tasks),
            "planner_validation_issues": repaired_issues,
            "planner_needs_repair": False,
            "planner_repair_attempted": True,
            "planner_error_reason": reason,
            "steps": [f"repair_plan:{reason}"],
        }

    return {
        "sub_tasks": tasks,
        "planner_raw_tasks": list(repaired.tasks),
        "planner_validation_issues": repaired_issues,
        "planner_needs_repair": False,
        "planner_repair_attempted": True,
        "planner_error_reason": "",
        "steps": ["repair_plan"],
    }


__all__ = ["repair_plan_node"]
