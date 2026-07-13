"""合规硬规则。"""

from __future__ import annotations

FORBIDDEN_PATTERNS = (
    "保证收益",
    "稳赚",
    "一定上涨",
    "必涨",
    "直接买入",
)


def find_rule_violations(text: str) -> list[str]:
    return [pattern for pattern in FORBIDDEN_PATTERNS if pattern in text]
