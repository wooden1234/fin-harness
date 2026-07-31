"""根据请求画像确定性推导 Root 任务所需能力。"""

from __future__ import annotations

from collections.abc import Iterable

from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.contracts import RequestProfile


_FINANCE_INTENT_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "concept_explain": ("faq",),
    "product_policy": ("faq",),
    "document_qa": ("pdf",),
    "structured_metric": ("financial_query",),
    "financial_research": ("financial_query",),
    "financial_analysis": ("financial_query",),
}

_FINANCE_QUERY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pdf", ("年报", "季报", "财报", "招股书", "上传的文档", "上传文件")),
    ("faq", ("什么是", "如何计算", "怎么计算", "规则是什么", "政策是什么")),
    (
        "financial_query",
        (
            "营收",
            "收入",
            "净利润",
            "毛利率",
            "净利率",
            "现金流",
            "市盈率",
            "市净率",
            "ROE",
            "ROA",
            "EPS",
            "财务数据",
        ),
    ),
)


def _registered_finance_capabilities() -> set[str]:
    return set(get_agent_spec("finance_agent").capabilities)


def _query_capability(query: str) -> str | None:
    for capability, markers in _FINANCE_QUERY_MARKERS:
        if any(marker in query for marker in markers):
            return capability
    return None


def resolve_finance_capabilities(
    profile: RequestProfile,
    *,
    allowed_capabilities: Iterable[str] = (),
) -> list[str]:
    """推导 Finance 任务能力，并与 Root 授予的能力范围求交。

    `finance_rag` 是入口画像中的聚合来源，不能直接当作执行能力；
    先用问题特征和意图细化，最后只返回注册且被授权的能力。
    """
    registered = _registered_finance_capabilities()
    allowed = set(allowed_capabilities) or registered
    candidates: list[str] = []

    query_capability = _query_capability(
        profile.normalized_query or profile.original_query
    )
    if query_capability:
        # 问题中的明确来源/操作信号优先于泛化意图，避免年报问题同时
        # 被声明为 pdf 和 financial_query，导致 Root 能力边界变宽。
        candidates.append(query_capability)
    else:
        for intent in profile.intents:
            candidates.extend(_FINANCE_INTENT_CAPABILITIES.get(intent, ()))

    # `none` 表示用户明确没有可用来源，不能回退到 Finance 默认能力。
    if "none" in profile.data_sources:
        candidates = []
    elif not candidates and "finance_rag" in profile.data_sources:
        # 保持旧版金融指标问题的兼容行为，同时避免把聚合来源写入 TaskSpec。
        candidates.append("financial_query")

    return list(dict.fromkeys(
        capability
        for capability in candidates
        if capability in registered and capability in allowed
    ))


__all__ = ["resolve_finance_capabilities"]
