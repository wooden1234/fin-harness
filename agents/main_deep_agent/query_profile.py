"""按问题复杂度选择 Main DeepAgent 的最小运行配置。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

MainQueryProfile = Literal[
    "simple_finance",
    "light_finance_analysis",
    "full_research",
]
PreferredOutputFormat = Literal["table", "markdown", "plain_text", ""]

_FINANCIAL_METRIC_MARKERS = {
    "net_profit": ("归母净利润", "扣非净利润", "净利润", "净利"),
    "revenue": ("营业收入", "营收", "收入"),
    "roe": ("roe", "净资产收益率"),
    "cash_flow": ("现金流", "经营活动现金"),
    "debt_ratio": ("负债率", "资产负债率"),
    "gross_margin": ("毛利率",),
    "net_margin": ("净利率",),
}

_FINANCE_MARKERS = (
    "营收", "营业收入", "净利润", "归母净利润", "扣非净利润", "毛利率",
    "净利率", "roe", "现金流", "负债率", "业绩预告", "财报", "年报",
    "半年报", "季报", "同比", "环比", "增幅", "增长多少",
)
_LIGHT_ANALYSIS_MARKERS = (
    "中枢", "上下限", "区间", "驱动因素", "变动原因", "增长原因",
    "不确定性", "风险因素", "简要说明", "简要分析",
)
_FULL_RESEARCH_MARKERS = (
    "对比", "比较", "行业", "板块", "估值", "投资价值", "是否值得",
    "持续性", "竞争格局", "产业链", "政策", "传闻", "研报", "目标价",
    "多家公司", "分别分析", "深度分析", "全面分析",
)
_MULTI_ENTITY_MARKERS = re.compile(r"(?:、|，|,|/|和|与|以及).{0,12}(?:公司|股份|集团)")


def classify_main_query_profile(query: str) -> MainQueryProfile:
    """仅将边界清晰的单公司财务题放入低成本通道。"""
    normalized = " ".join(str(query or "").lower().split())
    if not normalized or not any(marker in normalized for marker in _FINANCE_MARKERS):
        return "full_research"
    if any(marker in normalized for marker in _FULL_RESEARCH_MARKERS):
        return "full_research"
    if _MULTI_ENTITY_MARKERS.search(normalized):
        return "full_research"
    if any(marker in normalized for marker in _LIGHT_ANALYSIS_MARKERS):
        return "light_finance_analysis"
    return "simple_finance"


def required_financial_metrics(query: str) -> tuple[str, ...]:
    """提取问题明确要求的财务指标族，用于校验工具返回字段。"""
    normalized = " ".join(str(query or "").lower().split())
    return tuple(
        metric
        for metric, markers in _FINANCIAL_METRIC_MARKERS.items()
        if any(marker in normalized for marker in markers)
    )


def rewrite_fast_finance_query(query: str) -> str:
    """为快速通道补足容易被问财误解的业绩预告字段。"""
    normalized = " ".join(str(query or "").split()).strip()
    lowered = normalized.lower()
    if (
        "net_profit" in required_financial_metrics(normalized)
        and any(marker in lowered for marker in ("预计", "预告", "增长多少", "增幅"))
        and not any(marker in lowered for marker in ("增长率上限", "增长率下限"))
    ):
        return f"{normalized} 业绩预告净利润增长率下限 上限"
    return normalized


def _output_format_value(value: object) -> PreferredOutputFormat:
    if isinstance(value, Mapping):
        value = value.get("value", value.get("format", ""))
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return "table"
    if normalized == "markdown":
        return "markdown"
    if normalized == "plain_text":
        return "plain_text"
    return ""


def resolve_output_format_preference(
    query: str,
    *,
    memory_context: Mapping[str, object] | None = None,
    turn_preferences: Mapping[str, object] | None = None,
) -> PreferredOutputFormat:
    """按当前请求、本轮临时要求、长期记忆的顺序解析输出格式。"""
    normalized = "".join(str(query or "").lower().split())
    if any(marker in normalized for marker in ("不要表格", "不用表格", "无需表格", "别用表格", "纯文本")):
        return "plain_text"
    if any(marker in normalized for marker in ("用表格", "表格展示", "表格回答", "做成表格")):
        return "table"
    turn_value = _output_format_value(
        (turn_preferences or {}).get("preferred_output_format")
    )
    if turn_value:
        return turn_value
    return _output_format_value(
        (memory_context or {}).get("preferred_output_format")
    )


__all__ = [
    "MainQueryProfile",
    "PreferredOutputFormat",
    "classify_main_query_profile",
    "required_financial_metrics",
    "resolve_output_format_preference",
    "rewrite_fast_finance_query",
]
