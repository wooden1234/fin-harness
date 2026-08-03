"""工具统一执行入口。"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from typing import Any, Mapping

from harness.context import RunContext
from harness.policy import can_use_tool
from tools.core.base import ToolResult
from tools.core.registry import get_registered_tool


def _normalize_result(tool_id: str, data: Any) -> ToolResult:
    """把工具自身的失败响应收敛为统一 ToolResult。"""
    if isinstance(data, Mapping) and data.get("ok") is False:
        return ToolResult(
            tool_id=tool_id,
            ok=False,
            error=str(data.get("error") or "tool_execution_failed"),
            metadata={
                key: value
                for key, value in data.items()
                if key not in {"ok", "error", "data"}
            },
        )
    return ToolResult(tool_id=tool_id, ok=True, data=data)


async def execute_tool(
    tool_id: str,
    context: RunContext,
    *,
    arguments: dict[str, Any] | None = None,
    allowed_tool_ids: Collection[str] | None = None,
) -> ToolResult:
    """执行注册表中的工具，统一处理权限、超时、有限重试与返回结构。"""
    if allowed_tool_ids is not None and tool_id not in set(allowed_tool_ids):
        return ToolResult(tool_id=tool_id, ok=False, error="tool_not_allowed_for_agent")
    if not can_use_tool(context, tool_id):
        return ToolResult(tool_id=tool_id, ok=False, error="permission_denied")

    entry = get_registered_tool(tool_id)
    payload = arguments or {}
    if not isinstance(payload, dict):
        return ToolResult(tool_id=tool_id, ok=False, error="invalid_arguments")

    timeout_seconds = max(0.1, float(entry.spec.timeout_seconds))
    retry_count = (
        max(0, int(entry.spec.max_retries))
        if entry.spec.read_only
        else 0
    )

    for attempt in range(retry_count + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                data = await entry.handler(payload)
            return _normalize_result(tool_id, data)
        except TimeoutError:
            if attempt >= retry_count:
                return ToolResult(tool_id=tool_id, ok=False, error="tool_timeout")
        except (ConnectionError, OSError):
            if attempt >= retry_count:
                return ToolResult(
                    tool_id=tool_id,
                    ok=False,
                    error="tool_unavailable",
                )
        except (TypeError, ValueError):
            return ToolResult(tool_id=tool_id, ok=False, error="invalid_arguments")
        except Exception:  # noqa: BLE001
            return ToolResult(
                tool_id=tool_id,
                ok=False,
                error="tool_execution_failed",
            )

    return ToolResult(tool_id=tool_id, ok=False, error="tool_execution_failed")


__all__ = ["execute_tool"]
