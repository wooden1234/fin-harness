"""finance_agent planner 兼容聚合导出。

具体节点实现已拆到 plan_tasks / validate_plan / repair_plan / dispatch_workers。
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.finance_agent.planner.common import (
    is_schema_error as _is_schema_error,
    is_transient_api_error as _is_transient_api_error,
)
from agents.finance_agent.planner.dispatch_workers import (
    dispatch_workers_node,
    route_after_dispatch_workers,
    route_after_retrieval_worker,
    route_after_supervisor,
)
from agents.finance_agent.planner.plan_tasks import plan_tasks_node
from agents.finance_agent.planner.repair_plan import repair_plan_node
from agents.finance_agent.planner.validate_plan import (
    route_after_validate_plan,
    validate_plan_node,
)


async def supervisor_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """兼容旧入口：顺序执行 plan → validate → repair?。"""
    current: dict = dict(state)
    for node in (plan_tasks_node, validate_plan_node):
        current.update(await node(cast(FinAgentState, current), config))
    if route_after_validate_plan(cast(FinAgentState, current)) == "repair_plan":
        current.update(await repair_plan_node(cast(FinAgentState, current), config))
    return {
        key: value
        for key, value in current.items()
        if key
        in {
            "task_results",
            "citations",
            "summary",
            "sub_tasks",
            "planner_query",
            "planner_raw_tasks",
            "planner_validation_issues",
            "planner_needs_repair",
            "planner_repair_attempted",
            "planner_error_reason",
            "steps",
        }
    }


__all__ = [
    "_is_schema_error",
    "_is_transient_api_error",
    "dispatch_workers_node",
    "plan_tasks_node",
    "repair_plan_node",
    "route_after_dispatch_workers",
    "route_after_retrieval_worker",
    "route_after_supervisor",
    "route_after_validate_plan",
    "supervisor_node",
    "validate_plan_node",
]
