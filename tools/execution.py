"""工具统一执行入口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from harness.context import RunContext
from harness.policy import can_use_tool
from tools.base import ToolResult
from tools.registry import get_registered_tool


async def execute_tool(
    tool_id: str,
    context: RunContext,
    func: Callable[..., Awaitable[Any]] | None = None,
    /,
    *args: Any,
    **kwargs: Any,
) -> ToolResult:
    """执行受控工具，后续在这里补超时、重试、审计。

    - 传入 ``func``：直接执行该协程（兼容旧调用）
    - 不传 ``func``：从注册表取 ``langchain_tool.ainvoke(kwargs)``
    """
    if not can_use_tool(context, tool_id):
        return ToolResult(tool_id=tool_id, ok=False, error="permission_denied")
    try:
        if func is not None:
            data = await func(*args, **kwargs)
        else:
            entry = get_registered_tool(tool_id)
            if entry.langchain_tool is None:
                return ToolResult(
                    tool_id=tool_id,
                    ok=False,
                    error="tool_has_no_langchain_binding",
                )
            # LangChain tool 习惯吃 dict 入参
            payload = kwargs if kwargs else (args[0] if args else {})
            data = await entry.langchain_tool.ainvoke(payload)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool_id=tool_id, ok=False, error=str(exc))
    return ToolResult(tool_id=tool_id, ok=True, data=data)


__all__ = ["execute_tool"]
