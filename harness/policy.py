"""运行前策略检查。"""

from __future__ import annotations

from harness.context import RunContext


def pre_check(context: RunContext) -> None:
    """预留策略入口，后续接入权限、审批、用户同意。"""
    return None


def can_use_tool(context: RunContext, tool_id: str) -> bool:
    """按显式权限判定工具；缺少权限时默认拒绝。"""
    if not context.permissions:
        return False
    try:
        from tools.core.registry import get_tool_spec

        get_tool_spec(tool_id)
    except KeyError:
        return False
    return tool_id in context.permissions or "*" in context.permissions
