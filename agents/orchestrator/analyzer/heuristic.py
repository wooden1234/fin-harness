"""确定性请求画像：LLM Analyzer 的兜底实现。

选股与金融判定偏严；无法明确归类时一律走 general_agent。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents.orchestrator.contracts import RequestProfile

# 明确选股动作；单独的“筛选/选出”还需搭配股票对象，避免误伤“筛选基金”等。
_STOCK_DIRECT = ("选股", "荐股", "股票筛选")
_STOCK_ACTIONS = ("筛选", "选出", "挑选", "推荐几只", "挑几只", "找几只")
_STOCK_OBJECTS = ("股票", "A股", "个股", "沪深京")

# 明确金融事实/指标/文档信号；不含宽泛的“股票/新能源”。
_FINANCE_MARKERS = (
    "营收",
    "净利润",
    "毛利率",
    "净利率",
    "财报",
    "年报",
    "季报",
    "研报",
    "招股书",
    "公告",
    "市盈率",
    "市净率",
    "ROE",
    "ROA",
    "EPS",
    "财务",
    "资产负债表",
    "现金流量",
    "分红",
    "估值",
    "股价",
    "行情",
    "基金费率",
    "申购",
    "赎回",
    "交易规则",
    "T+1",
)

_COMPOUND_MARKERS = ("分析", "比较", "风险", "原因", "前三", "对比")


def latest_query(state: dict[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


def _is_stock_screening(query: str) -> bool:
    if any(marker in query for marker in _STOCK_DIRECT):
        return True
    has_action = any(marker in query for marker in _STOCK_ACTIONS)
    has_object = any(marker in query for marker in _STOCK_OBJECTS)
    return has_action and has_object


def _is_finance(query: str) -> bool:
    return any(marker in query for marker in _FINANCE_MARKERS)


def heuristic_profile(query: str) -> RequestProfile:
    """严格启发式画像：stock / finance 明确命中，否则 general。"""
    if not query:
        return RequestProfile(
            original_query="",
            normalized_query="",
            complexity="simple",
            missing_fields=["query"],
        )

    if _is_stock_screening(query):
        is_compound = any(marker in query for marker in _COMPOUND_MARKERS)
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["stock_screening"] + (["financial_analysis"] if is_compound else []),
            complexity="compound" if is_compound else "single_capability",
            freshness_required=True,
            preferred_agent="stock_screening_agent",
        )

    if _is_finance(query):
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["financial_research"],
            complexity="single_capability",
            preferred_agent="finance_agent",
        )

    return RequestProfile(
        original_query=query,
        normalized_query=query,
        intents=["general_chat"],
        complexity="simple",
        preferred_agent="general_agent",
    )


__all__ = ["heuristic_profile", "latest_query"]
