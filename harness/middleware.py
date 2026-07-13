"""Harness 中间件占位。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.context import RunContext


async def run_with_middlewares(
    context: RunContext,
    handler: Callable[[RunContext], Any],
) -> Any:
    """预留 PII、注入防护、限流和异常兜底的统一入口。"""
    return await handler(context)
