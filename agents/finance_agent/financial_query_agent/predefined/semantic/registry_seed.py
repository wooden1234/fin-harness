"""语义注册表静态种子数据（第一版不落库）。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricDefinition,
)

CANONICAL_METRICS: dict[str, CanonicalMetricDefinition] = {
    "REVENUE": CanonicalMetricDefinition(
        code="REVENUE",
        name="营业收入",
        description="合并报表营业收入",
    ),
    "NET_INCOME_ATTR_PARENT": CanonicalMetricDefinition(
        code="NET_INCOME_ATTR_PARENT",
        name="归母净利润",
        description="归属于上市公司股东的净利润",
    ),
    "OPERATING_PROFIT": CanonicalMetricDefinition(
        code="OPERATING_PROFIT",
        name="营业利润",
        description="营业利润",
    ),
    "RND_EXPENSE": CanonicalMetricDefinition(
        code="RND_EXPENSE",
        name="研发费用",
        description="研发费用",
    ),
    "OPERATING_CASHFLOW_NET": CanonicalMetricDefinition(
        code="OPERATING_CASHFLOW_NET",
        name="经营现金流",
        description="经营活动产生的现金流量净额",
    ),
}

GLOBAL_ALIASES: dict[str, str] = {
    "营业收入": "REVENUE",
    "营收": "REVENUE",
    "收入": "REVENUE",
    "营业额": "REVENUE",
    "主营业务收入": "REVENUE",
    "净利润": "NET_INCOME_ATTR_PARENT",
    "归母净利润": "NET_INCOME_ATTR_PARENT",
    "归母净利": "NET_INCOME_ATTR_PARENT",
    "净利": "NET_INCOME_ATTR_PARENT",
    "股东净利润": "NET_INCOME_ATTR_PARENT",
    "归属于上市公司股东的净利润": "NET_INCOME_ATTR_PARENT",
    "营业利润": "OPERATING_PROFIT",
    "研发费用": "RND_EXPENSE",
    "研发": "RND_EXPENSE",
    "研发投入": "RND_EXPENSE",
    "经营活动产生的现金流量净额": "OPERATING_CASHFLOW_NET",
    "经营现金流": "OPERATING_CASHFLOW_NET",
    "现金流": "OPERATING_CASHFLOW_NET",
    "经营现金流净额": "OPERATING_CASHFLOW_NET",
}

COMPANY_OVERRIDES: dict[str, dict[str, tuple[str, ...]]] = {
    "Tencent": {
        "REVENUE": ("收入",),
        "NET_INCOME_ATTR_PARENT": ("本公司權益持有人應佔盈利",),
    },
    "CATL": {
        "REVENUE": ("营业收入(千元)", "营业收入"),
        "NET_INCOME_ATTR_PARENT": ("归属于上市公司股东的净利润",),
    },
}

def resolve_canonical_code(metric_text: str) -> str | None:
    """将用户指标词映射到 canonical metric code。"""
    cleaned = metric_text.strip()
    if not cleaned:
        return None
    if cleaned in GLOBAL_ALIASES:
        return GLOBAL_ALIASES[cleaned]
    normalized = cleaned.replace(" ", "").lower()
    for alias, code in GLOBAL_ALIASES.items():
        alias_norm = alias.replace(" ", "").lower()
        if normalized == alias_norm or normalized in alias_norm or alias_norm in normalized:
            return code
    return None


def company_metric_names(company_key: str, canonical_code: str) -> list[str]:
    """返回公司级覆盖指标名列表，若无覆盖则返回 canonical 默认名。"""
    overrides = COMPANY_OVERRIDES.get(company_key, {}).get(canonical_code)
    if overrides:
        return list(overrides)
    definition = CANONICAL_METRICS.get(canonical_code)
    return [definition.name] if definition else []


__all__ = [
    "CANONICAL_METRICS",
    "COMPANY_OVERRIDES",
    "GLOBAL_ALIASES",
    "company_metric_names",
    "resolve_canonical_code",
]
