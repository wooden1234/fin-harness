"""运行前策略检查。"""

from __future__ import annotations

from harness.context import RunContext


def pre_check(context: RunContext) -> None:
    """预留策略入口，后续接入权限、审批、用户同意。"""
    return None


def can_use_tool(context: RunContext, tool_id: str) -> bool:
    """默认允许只读工具，后续按 ToolSpec 风险等级收紧。"""
    if not context.permissions:
        return True
    return tool_id in context.permissions or "*" in context.permissions
