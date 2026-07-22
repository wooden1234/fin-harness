"""validate_plan 节点：确定性校验 planner 输出。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.states import FinAgentState
from agents.finance_agent.planner.common import assign_task_ids, logger
from agents.finance_agent.planner.validate import validate_and_normalize_tasks


async def validate_plan_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """确定性校验 planner 输出，并决定是否进入 repair。"""
    del config
    raw_tasks = list(state.get("planner_raw_tasks") or [])
    preset_issues = list(state.get("planner_validation_issues") or [])
    if state.get("planner_error_reason") in {"empty_query", "api_error", "unexpected_error"}:
        return {
            "sub_tasks": [],
            # 空计划不应继承上一轮并行 worker 的结果。
            "task_results": Overwrite([]),
            "planner_needs_repair": False,
            "planner_validation_issues": preset_issues,
            "steps": ["validate_plan:skip"],
        }
    if any(str(issue).startswith("schema_error:") for issue in preset_issues):
        return {
            "sub_tasks": [],
            "task_results": Overwrite([]),
            "planner_validation_issues": preset_issues,
            "planner_needs_repair": True,
            "steps": ["validate_plan:needs_repair"],
        }

    validation = validate_and_normalize_tasks(raw_tasks)
    issues = preset_issues + validation.issues
    if validation.needs_repair:
        logger.warning(
            "planner validation issues={}, routing to repair",
            issues,
        )
        return {
            "sub_tasks": validation.tasks,
            "planner_validation_issues": issues,
            "planner_needs_repair": True,
            "steps": ["validate_plan:needs_repair"],
        }

    tasks = assign_task_ids(validation.tasks)
    if not tasks:
        reason = "unclassifiable" if not issues else "invalid_after_validate"
        logger.info(
            "planner empty tasks reason={} issues={}",
            reason,
            issues,
        )
        return {
            "sub_tasks": [],
            # reducer 默认是追加，必须显式覆盖才能清理旧结果。
            "task_results": Overwrite([]),
            "planner_validation_issues": issues,
            "planner_needs_repair": False,
            "planner_error_reason": reason,
            "steps": [f"validate_plan:{reason}"],
        }

    logger.info(
        "planner tasks={} intents={} types={} issues={}",
        len(tasks),
        [t.intent for t in tasks],
        [t.type for t in tasks],
        issues,
    )
    return {
        "sub_tasks": tasks,
        "planner_validation_issues": issues,
        "planner_needs_repair": False,
        "planner_error_reason": "",
        "steps": ["validate_plan"],
    }


def route_after_validate_plan(state: FinAgentState) -> str:
    """校验后按需进入 repair，否则进入证据链解析。"""
    if bool(state.get("planner_needs_repair")) and not bool(state.get("planner_repair_attempted")):
        return "repair_plan"
    return "resolve_evidence"


__all__ = ["route_after_validate_plan", "validate_plan_node"]
