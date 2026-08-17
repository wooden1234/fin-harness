"""CJK 感知粗测。"""

from __future__ import annotations

from collections.abc import Sequence

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
