"""长期记忆冲突检测。"""

from __future__ import annotations

from typing import Any


def values_conflict(current: Any, incoming: Any) -> bool:
    """仅当两个规范化值不同才视为冲突。"""
    return current != incoming
