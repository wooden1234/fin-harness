"""Finance Planner 的领域授权范围读取与展示。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def domain_scope_from_state(state: Mapping[str, Any]):
    """读取并校验 Root 下发的 Scope；缺失表示兼容旧 V1 调用。"""
    raw = state.get("domain_planning_scope")
    if raw is None:
        task_input = state.get("task_input")
        if isinstance(task_input, Mapping):
            raw = task_input.get("domain_planning_scope")
    if raw is None:
        return None

    # 延迟导入，避免 Finance State 初始化时反向加载整个 Orchestrator 包。
    from agents.orchestrator.contracts import DomainPlanningScope

    if isinstance(raw, DomainPlanningScope):
        return raw
    return DomainPlanningScope.model_validate(raw)


def scope_prompt(scope: object | None) -> str:
    """把 Scope 转换为 Planner 可读提示；代码校验仍是最终安全边界。"""
    if scope is None:
        return "领域授权范围：旧版兼容模式。"
    capabilities = ", ".join(getattr(scope, "allowed_capabilities", [])) or "无"
    intents = ", ".join(getattr(scope, "allowed_intents", [])) or "无"
    return (
        f"领域授权能力：{capabilities}\n"
        f"允许业务意图：{intents}\n"
        f"最多子任务数：{getattr(scope, 'max_subtasks', 4)}\n"
        "不得规划或降级到授权范围之外的证据渠道。"
    )


__all__ = ["domain_scope_from_state", "scope_prompt"]
