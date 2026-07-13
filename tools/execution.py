"""工具统一执行入口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from harness.context import RunContext
from harness.policy import can_use_tool
from tools.base import ToolResult


async def execute_tool(
    tool_id: str,
    context: RunContext,
    func: Callable[..., Awaitable[Any]],
    /,
    *args: Any,
    **kwargs: Any,
) -> ToolResult:
    """执行受控工具，后续在这里补超时、重试、审计。"""
    if not can_use_tool(context, tool_id):
        return ToolResult(tool_id=tool_id, ok=False, error="permission_denied")
    try:
        data = await func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool_id=tool_id, ok=False, error=str(exc))
    return ToolResult(tool_id=tool_id, ok=True, data=data)
