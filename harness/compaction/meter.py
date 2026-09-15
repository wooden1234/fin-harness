"""CJK 感知粗测。"""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Mapping

from harness.session.surface import SurfaceMessage


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    rest = max(0, len(text) - cjk)
    return cjk + rest // 4


def surface_tokens(messages: Sequence[SurfaceMessage]) -> int:
    total = 0
    for item in messages:
        total += 8 + estimate_tokens(item.content)
        if item.tool_calls:
            total += 8 * len(item.tool_calls)
    return total


def request_tokens(
    *,
    system: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    """无 provider tokenizer 时的保守完整请求估算。"""
    total = 16 + estimate_tokens(system)
    for message in messages:
        total += 8 + estimate_tokens(str(message.get("content") or ""))
        calls = message.get("tool_calls") or []
        if calls:
            total += 8 + estimate_tokens(
                json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str)
            )
    if tools:
        total += 16 + estimate_tokens(
            json.dumps(list(tools), ensure_ascii=False, sort_keys=True, default=str)
        )
    return total


async def count_request_tokens(
    llm: Any,
    *,
    system: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    """优先使用 adapter 计数；同步和异步 adapter 均可。"""
    counter = getattr(llm, "count_request_tokens", None)
    if callable(counter):
        try:
            value = counter(system=system, messages=messages, tools=tools)
            if hasattr(value, "__await__"):
                value = await value
            counted = int(value)
            if counted >= 0:
                return counted
        except Exception:  # noqa: BLE001 - 计数失败必须安全回退
            pass
    return request_tokens(system=system, messages=messages, tools=tools)
