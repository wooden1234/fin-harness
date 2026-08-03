"""投资动作敏感识别与输出合规策略。"""

from __future__ import annotations


INVESTMENT_ACTION_MARKERS = (
    "能买吗",
    "能不能买",
    "介入",
    "加仓",
    "低吸",
    "止损",
    "目标价",
    "买点",
    "卖点",
    "仓位",
)


def is_investment_action_sensitive(query: str) -> bool:
    """判断问题是否包含需要强化约束的投资动作表达。"""
    normalized = query.replace(" ", "").lower()
    return any(marker.lower() in normalized for marker in INVESTMENT_ACTION_MARKERS)


__all__ = ["INVESTMENT_ACTION_MARKERS", "is_investment_action_sensitive"]

