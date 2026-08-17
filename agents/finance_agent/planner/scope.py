"""Finance Planner 的领域授权范围读取与展示。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def domain_scope_from_state(state: Mapping[str, Any]):
    raw = state.get("domain_planning_scope")
    if raw is None:
        task_input = state.get("task_input")
        if isinstance(task_input, Mapping):
            raw = task_input.get("domain_planning_scope")
    return raw


def scope_prompt(scope: object | None) -> str:
    if scope is None:
        return "领域授权范围：旧版兼容模式。"
    capabilities = ", ".join(getattr(scope, "allowed_capabilities", []) or []) or "无"
    intents = ", ".join(getattr(scope, "allowed_intents", []) or []) or "无"
    return (
        f"领域授权能力：{capabilities}\n"
        f"允许业务意图：{intents}\n"
        f"最多子任务数：{getattr(scope, 'max_subtasks', 4)}\n"
        "不得规划或降级到授权范围之外的证据渠道。"
    )
