"""注册工具的统一受限执行入口。"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.context import RunContext
from tools import execute_tool


async def execute_registered_tool(
    *,
    tool_id: str,
    run_context: RunContext,
    arguments: dict[str, Any],
    allowed_tool_ids: frozenset[str],
    timeout_seconds: float,
):
    """在工具级超时和白名单内执行一次注册工具。"""
    async with asyncio.timeout(timeout_seconds):
        return await execute_tool(
            tool_id,
            run_context,
            arguments=arguments,
            allowed_tool_ids=allowed_tool_ids,
        )


__all__ = ["execute_registered_tool"]

