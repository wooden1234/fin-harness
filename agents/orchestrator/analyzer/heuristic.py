"""确定性请求画像：LLM Analyzer 的兜底实现。

选股与金融判定偏严；无法明确归类时一律走 general_agent。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents.orchestrator.analyzer.schema import AnalyzerConstraints, AnalyzerOutput
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
    # 评级/目标价 markers 需先于裸词“研报”匹配，避免“研报评级”被误判为研报搜索。
    ("iwencai.rating.query", ("机构评级", "研报评级", "目标价", "业绩预测", "ESG")),
    ("iwencai.report.search", ("研报", "研报搜索", "研究报告", "券商研报")),
)

_MARKET_TOOL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("iwencai.fund.screen", ("基金筛选", "公募基金", "基金经理", "基金持仓")),
    ("iwencai.industry.query", ("行业估值", "行业排名", "板块排名", "行业行情")),
    ("iwencai.index.query", ("指数行情", "指数点位", "沪深300", "上证指数", "创业板指")),
    ("iwencai.market.query", ("股价", "行情", "涨跌幅", "成交量", "资金流向", "技术指标")),
)

_COMPOUND_MARKERS = ("分析", "比较", "风险", "原因", "前三", "对比")
_DEEP_RESEARCH_MARKERS = ("深度研究", "全面研究", "综合研究", "系统分析")
_OPEN_RESEARCH_SCOPE_MARKERS = (
    "白酒龙头",
    "新能源行业",
    "半导体行业",
    "银行板块",
    "券商板块",
    "龙头",
    "行业",
    "板块",
    "概念股",
    "白酒",
    "新能源",
    "半导体",
    "银行股",
    "券商股",
)
_OPEN_RESEARCH_GOAL_MARKERS = (
    "表现怎么样",
    "表现如何",
    "发展趋势",
    "行业前景",
    "主要风险",
    "竞争力",
    "投资逻辑",
    "为什么上涨",
    "为什么下跌",
    "为什么涨",
    "为什么跌",
    "未来怎么样",
    "未来如何",
    "怎么看",
)
_COMPUTE_CONTEXT_MARKERS = ("刚才", "上述", "候选集", "候选股票", "上一步")
_COMPUTE_ACTION_MARKERS = ("过滤", "排序", "最高", "最低", "前", "后")
_DOCUMENT_MARKERS = ("年报", "季报", "财报", "招股书", "白皮书", "政策文件", "这份文档")
_STRUCTURED_METRIC_MARKERS = (
    "营收", "营业收入", "净利润", "毛利率", "净利率", "现金流", "市盈率", "市净率", "ROE", "ROA", "EPS"
)
_PRODUCT_POLICY_MARKERS = (
    "报销", "付款", "审批", "预算", "内控", "发票", "申购费", "赎回费", "办理条件"
)
_CONCEPT_MARKERS = ("什么是", "如何计算", "怎么计算", "规则是什么", "制度是什么")
_COMPARISON_MARKERS = ("比较", "对比", "相比", "分别")
_UNDERSPECIFIED_REQUESTS = frozenset(
    {
        "帮我看看",
        "帮我分析一下",
        "看看",
        "分析一下",
        "怎么样",
        "给点意见",
        "给个意见",
        "详细说说",
        "展开说说",
        "有啥建议",
        "有什么建议",
        "帮忙判断一下",
    }
)
_UNDERSPECIFIED_ACTION_PARTS = (
    "请",
    "帮我",
    "帮忙",
    "一下",
    "详细",
    "具体",
    "看看",
    "分析",
    "判断",
    "说说",
    "展开",
    "给点意见",
    "给个意见",
    "有啥建议",
    "有什么建议",
)


def _is_underspecified(query: str) -> bool:
    normalized = "".join(query.split()).rstrip("？?！!。")
    if normalized in _UNDERSPECIFIED_REQUESTS:
        return True
    remainder = normalized
    for part in _UNDERSPECIFIED_ACTION_PARTS:
        remainder = remainder.replace(part, "")
    return remainder in {"", "这个", "那个", "它", "该公司", "这家公司"}


def latest_query(state: dict[str, Any]) -> str:
    rewritten_query = str(state.get("rewritten_query") or "").strip()
    rewrite_status = str(state.get("rewrite_status") or "").strip()
    if rewritten_query and rewrite_status in {"rewrite", "passthrough"}:
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


def _open_research_entity(query: str) -> str:
    if not any(marker in query for marker in _OPEN_RESEARCH_GOAL_MARKERS):
        return ""
    return next(
        (marker for marker in _OPEN_RESEARCH_SCOPE_MARKERS if marker in query),
        "",
    )


def heuristic_profile(
    query: str,
    *,
    candidate_set_id: str | None = None,
) -> RequestProfile:
    """严格启发式画像：先拦截不可执行请求，再进行领域与能力映射。"""
    from agents.orchestrator.analyzer.validate import validate_and_normalize

    def build(
        intents: list[str],
        *,
        freshness: bool = False,
        entities: list[str] | None = None,
        constraints: AnalyzerConstraints | None = None,
        missing_fields: list[str] | None = None,
    ) -> RequestProfile:
        raw = AnalyzerOutput(
            normalized_query=query,
            intents=intents,
            freshness_required=freshness,
            entities=entities or [],
            constraints=constraints or AnalyzerConstraints(),
            missing_fields=missing_fields or [],
        )
        return validate_and_normalize(raw, original_query=query).profile

    if not query:
        return build(["clarify"], missing_fields=["query"])

    if _is_underspecified(query):
        return build(["clarify"], missing_fields=["query_target"])

    open_research_entity = _open_research_entity(query)
    if any(marker in query for marker in _COMPARISON_MARKERS):
        return build(
            ["entity_comparison"],
            freshness=True,
            constraints=AnalyzerConstraints(entity_scope_type="explicit_group"),
        )
    if open_research_entity or any(
        marker in query for marker in (*_DEEP_RESEARCH_MARKERS, *_OPEN_RESEARCH_GOAL_MARKERS)
    ):
        return build(
            ["open_research"],
            freshness=True,
            entities=[open_research_entity] if open_research_entity else [],
            constraints=AnalyzerConstraints(
                entity_scope_type="dynamic_group" if open_research_entity else "single"
            ),
        )

    if _is_market_compute(query):
        return build(
            ["candidate_compute"],
            constraints=AnalyzerConstraints(candidate_set_id=candidate_set_id),
            missing_fields=[] if candidate_set_id else ["candidate_set_id"],
        )

    if _is_stock_screening(query):
        is_compound = any(marker in query for marker in _COMPOUND_MARKERS)
        if is_compound:
            return build(["stock_screening", "open_research"], freshness=True)
        return build(["stock_screening"], freshness=True)

    research_tool_id = research_tool_for_query(query)
    if research_tool_id and any(marker in query for marker in ("查询", "查找", "找一下", "搜索")):
        return build(["research_search"], freshness=True)

    market_tool_id = market_tool_for_query(query)
    if market_tool_id:
        return build(["market_query"], freshness=True)

    if any(marker in query for marker in _DOCUMENT_MARKERS) or research_tool_id:
        return build(["document_qa"])

    if any(marker in query for marker in _PRODUCT_POLICY_MARKERS):
        return build(["product_policy"])

    if any(marker in query for marker in _CONCEPT_MARKERS):
        return build(["concept_explain"])

    if any(marker in query for marker in _STRUCTURED_METRIC_MARKERS):
        return build(["structured_metric"])

    if _is_finance(query):
        return build(["concept_explain"])

    return build(["general_chat"])


__all__ = [
    "heuristic_profile",
    "latest_query",
    "market_tool_for_query",
    "research_tool_for_query",
]
