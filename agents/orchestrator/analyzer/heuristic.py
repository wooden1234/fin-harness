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

_RESEARCH_TOOL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("iwencai.announcement.search", ("公告", "回购", "分红派息", "资产重组")),
    ("iwencai.report.search", ("研报搜索", "研究报告", "券商研报")),
    ("iwencai.rating.query", ("机构评级", "研报评级", "目标价", "业绩预测", "ESG")),
)

_MARKET_TOOL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("iwencai.fund.screen", ("基金筛选", "公募基金", "基金经理", "基金持仓")),
    ("iwencai.industry.query", ("行业估值", "行业排名", "板块排名", "行业行情")),
    ("iwencai.index.query", ("指数行情", "指数点位", "沪深300", "上证指数", "创业板指")),
    ("iwencai.market.query", ("股价", "行情", "涨跌幅", "成交量", "资金流向", "技术指标")),
)

_COMPOUND_MARKERS = ("分析", "比较", "风险", "原因", "前三", "对比")
_DEEP_RESEARCH_MARKERS = ("深度研究", "全面研究", "综合研究", "系统分析")
_COMPUTE_CONTEXT_MARKERS = ("刚才", "上述", "候选集", "候选股票", "上一步")
_COMPUTE_ACTION_MARKERS = ("过滤", "排序", "最高", "最低", "前", "后")


def latest_query(state: dict[str, Any]) -> str:
    rewritten_query = str(state.get("rewritten_query") or "").strip()
    rewrite_status = str(state.get("rewrite_status") or "").strip()
    if rewritten_query and rewrite_status in {"success", "passthrough"}:
        return rewritten_query
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


def market_tool_for_query(query: str) -> str | None:
    """根据明确的数据类型选择问财 Tool；不明确时返回空。"""
    for tool_id, markers in _MARKET_TOOL_MARKERS:
        if any(marker in query for marker in markers):
            return tool_id
    return None


def research_tool_for_query(query: str) -> str | None:
    """根据公开文档类型选择研究 Tool；不明确时返回空。"""
    for tool_id, markers in _RESEARCH_TOOL_MARKERS:
        if any(marker in query for marker in markers):
            return tool_id
    return None


def _is_market_compute(query: str) -> bool:
    return any(marker in query for marker in _COMPUTE_CONTEXT_MARKERS) and any(
        marker in query for marker in _COMPUTE_ACTION_MARKERS
    )


def heuristic_profile(query: str) -> RequestProfile:
    """严格启发式画像：stock / finance 明确命中，否则 general。"""
    if not query:
        return RequestProfile(
            original_query="",
            normalized_query="",
            complexity="simple",
            missing_fields=["query"],
        )

    if any(marker in query for marker in _DEEP_RESEARCH_MARKERS):
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["deep_research"],
            complexity="compound",
            data_sources=["market", "research", "finance_rag"],
            operation_type="deep_research",
            freshness_required=True,
            preferred_agent="research_workflow",
        )

    if _is_market_compute(query):
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["market_compute"],
            complexity="single_capability",
            data_sources=["upstream_data"],
            operation_type="compute",
            preferred_agent="market.compute",
        )

    if _is_stock_screening(query):
        is_compound = any(marker in query for marker in _COMPOUND_MARKERS)
        if is_compound:
            return RequestProfile(
                original_query=query,
                normalized_query=query,
                intents=["stock_screening", "financial_analysis", "deep_research"],
                complexity="compound",
                data_sources=["market", "research", "finance_rag"],
                operation_type="deep_research",
                freshness_required=True,
                preferred_agent="research_workflow",
            )
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["stock_screening"],
            complexity="single_capability",
            data_sources=["market"],
            operation_type="acquire",
            freshness_required=True,
            preferred_agent="stock_screening_agent",
        )

    research_tool_id = research_tool_for_query(query)
    if research_tool_id:
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["research_search"],
            complexity="single_capability",
            data_sources=["research"],
            operation_type="retrieve",
            freshness_required=True,
            constraints={"research_tool_id": research_tool_id},
            preferred_agent="research_retrieval_workflow",
        )

    market_tool_id = market_tool_for_query(query)
    if market_tool_id:
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["market_query"],
            complexity="single_capability",
            data_sources=["market"],
            operation_type="acquire",
            freshness_required=True,
            constraints={"market_tool_id": market_tool_id},
            preferred_agent="market_acquisition_workflow",
        )

    if _is_finance(query):
        return RequestProfile(
            original_query=query,
            normalized_query=query,
            intents=["financial_research"],
            complexity="single_capability",
            data_sources=["finance_rag"],
            operation_type="analyze",
            preferred_agent="finance_agent",
        )

    return RequestProfile(
        original_query=query,
        normalized_query=query,
        intents=["general_chat"],
        complexity="simple",
        data_sources=["none"],
        operation_type="answer",
        preferred_agent="general_agent",
    )


__all__ = [
    "heuristic_profile",
    "latest_query",
    "market_tool_for_query",
    "research_tool_for_query",
]
