"""编排层执行档位：规则只处理高置信请求，灰区交给 LLM。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage

from agents.general_agent.weather_direct import parse_weather_request

ExecutionLane = Literal["general", "deep", "uncertain"]
RuleLaneRoute = Literal["resolved", "uncertain"]

# 金融强信号：命中则进 DeepAgent（不含单独「天气」）。
_FINANCE_MARKERS = (
    "营收", "净利润", "毛利率", "净利率", "财报", "年报", "季报", "财季",
    "研报", "招股书", "公告", "市盈率", "市净率", "ROE", "ROA", "EPS",
    "财务", "资产负债表", "现金流量", "分红", "估值", "股价", "行情",
    "基金费率", "申购", "赎回", "交易规则", "T+1", "超预期", "一致预期",
    "指引", "同比", "增速", "选股", "荐股", "A股", "个股", "港股", "美股",
    "机构评级", "目标价", "涨跌幅", "成交量", "资金流向", "板块", "概念股",
    "iwencai", "问财",
)
_RESEARCH_MARKERS = (
    "深度研究", "全面研究", "综合研究", "系统分析", "投资逻辑", "主要风险",
    "行业前景", "发展趋势", "为什么涨", "为什么跌", "为什么上涨", "为什么下跌",
)
_COMPARISON_MARKERS = ("比较", "对比", "相比", "versus", " vs ")
_GENERAL_GREETING_MARKERS = (
    "你好", "您好", "嗨", "hello", "hi", "嘿", "早上好", "中午好", "晚上好",
    "谢谢", "多谢", "再见", "拜拜", "在吗", "你是谁", "你叫什么",
)
_GENERAL_CHAT_MARKERS = (
    "聊聊天", "闲聊", "讲个笑话", "推荐电影", "推荐餐厅", "心情",
    "帮我翻译", "写一首",
)
_WEATHER_MARKERS = ("天气", "气温", "下雨", "降雨")
_CONCEPT_EXPLANATION_MARKERS = (
    "什么是",
    "何为",
    "定义",
    "概念",
    "是什么意思",
    "定义是什么",
    "概念是什么",
    "如何理解",
    "怎么理解",
    "如何计算",
    "怎么计算",
    "规则是什么",
    "制度是什么",
)
_EXTERNAL_FACT_REQUEST_MARKERS = (
    "最新",
    "当前",
    "今天",
    "多少",
    "查询",
    "查一下",
    "数据",
    "比较",
    "对比",
    "行情",
    "走势",
    "风险",
    "原因",
    "分析",
)


def latest_user_query(state: dict[str, Any]) -> str:
    rewritten = str(state.get("rewritten_query") or "").strip()
    if rewritten and str(state.get("rewrite_status") or "") in {"rewrite", "passthrough"}:
        return rewritten
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


def _has_finance_signal(query: str) -> bool:
    lowered = query.lower()
    if any(marker.lower() in lowered for marker in _FINANCE_MARKERS):
        return True
    if any(marker in query for marker in _RESEARCH_MARKERS):
        return True
    if any(marker in query for marker in _COMPARISON_MARKERS) and any(
        token in query for token in ("公司", "股", "和", "与", "及")
    ):
        return True
    return False


def _is_concept_explanation(query: str) -> bool:
    """识别无需外部事实的概念解释，不绑定具体金融术语。"""
    if not any(marker in query for marker in _CONCEPT_EXPLANATION_MARKERS):
        return False
    return not any(marker in query for marker in _EXTERNAL_FACT_REQUEST_MARKERS)


def _is_explicit_general(query: str) -> bool:
    normalized = "".join(query.split()).strip("？?！!。. ")
    if not normalized:
        return True
    lowered = normalized.lower()
    if any(marker.lower() in lowered for marker in _GENERAL_GREETING_MARKERS):
        # 「你好，帮我看看茅台股价」仍算金融
        if _has_finance_signal(query):
            return False
        return True
    if any(marker in query for marker in _WEATHER_MARKERS):
        return not _has_finance_signal(query)
    if any(marker in query for marker in _GENERAL_CHAT_MARKERS):
        return not _has_finance_signal(query)
    if len(normalized) <= 6 and not _has_finance_signal(query):
        if normalized in {"在", "嗯", "好的", "哦", "哈哈", "收到"}:
            return True
    return False


def classify_execution_lane(query: str) -> ExecutionLane:
    """规则定档：只返回高置信结果，无法确定时返回 uncertain。"""
    text = str(query or "").strip()
    if not text:
        return "general"
    if parse_weather_request(text) is not None:
        return "general"
    if _is_concept_explanation(text):
        return "general"
    if _has_finance_signal(text):
        return "deep"
    if _is_explicit_general(text):
        return "general"
    return "uncertain"


async def classify_execution_lane_node(state: dict[str, Any]) -> dict[str, Any]:
    """写入 execution_lane；普通档同时设 route=general 供终答透传。"""
    from app.core.logger import get_logger

    logger = get_logger(service="execution_lane")
    query = latest_user_query(state)
    lane = classify_execution_lane(query)
    logger.info(
        "execution_lane={} query={}",
        lane,
        " ".join(query.split())[:120],
    )
    update: dict[str, Any] = {
        "execution_lane": lane,
        "steps": [f"orchestrator:execution_lane:{lane}"],
    }
    if lane in {"general", "deep"}:
        update["route"] = lane
    return update


def route_after_rule_lane(state: dict[str, Any]) -> RuleLaneRoute:
    """高置信规则结果直接继续，灰区进入 LLM 档位解析。"""
    if str(state.get("execution_lane") or "").strip() == "uncertain":
        return "uncertain"
    return "resolved"


def route_after_execution_lane(state: dict[str, Any]) -> ExecutionLane:
    lane = str(state.get("execution_lane") or "").strip()
    if lane == "general":
        return "general"
    # 兼容：若上游节点类型注解收窄了通道，仍可用 route 兜底。
    if str(state.get("route") or "").strip() == "general":
        return "general"
    return "deep"


__all__ = [
    "ExecutionLane",
    "classify_execution_lane",
    "classify_execution_lane_node",
    "latest_user_query",
    "route_after_execution_lane",
    "route_after_rule_lane",
]
