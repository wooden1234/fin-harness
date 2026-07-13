"""白名单 SQL 模板描述，供 tool_selection prompt 与 catalog 使用。"""

from __future__ import annotations

EXACT_METRIC_LOOKUP = "exact_metric_lookup"
LATEST_METRIC_LOOKUP = "latest_metric_lookup"
COMPARE_METRIC_LOOKUP = "compare_metric_lookup"
TREND_METRIC_LOOKUP = "trend_metric_lookup"

# 当前 predefined 仅允许的已批准 canonical 口径。
APPROVED_CANONICAL_SCOPE_TEXT = (
    "当前白名单仅覆盖已批准口径：营业收入(REVENUE)、"
    "归属于上市公司股东的净利润(NET_INCOME_ATTR_PARENT)、"
    "营业利润(OPERATING_PROFIT)、"
    "经营活动产生的现金流量净额(OPERATING_CASHFLOW_NET)、"
    "研发费用(RND_EXPENSE)。"
    " 其他指标不要选 predefined，直接走 text_to_sql。"
)

VALID_TEMPLATE_IDS: frozenset[str] = frozenset(
    {
        EXACT_METRIC_LOOKUP,
        LATEST_METRIC_LOOKUP,
        COMPARE_METRIC_LOOKUP,
        TREND_METRIC_LOOKUP,
    }
)

QUERY_DESCRIPTIONS: dict[str, str] = {
    EXACT_METRIC_LOOKUP: (
        "单公司 + 单年份 + 单指标精确查数。"
        "仅用于已批准口径内的年度事实。"
        "适用于用户询问某公司在特定年份的某项财务指标。"
        "示例：宁德时代 2024 年营业收入是多少？"
    ),
    LATEST_METRIC_LOOKUP: (
        "单公司 + 单指标 + 最新一期查数，用户未指定年份。"
        "仅用于已批准口径内的最新已发布年度。"
        "适用于用户询问某公司最近/最新/当前的某项指标。"
        "示例：腾讯最新归母净利润是多少？"
    ),
    COMPARE_METRIC_LOOKUP: (
        "多公司同一指标对比查询。"
        "仅用于已批准口径内的横向对比，不支持同一公司多指标对比。"
        "适用于用户对比两家及以上公司的同一财务指标。"
        "示例：宁德时代和腾讯 2024 年营业收入对比"
    ),
    TREND_METRIC_LOOKUP: (
        "单公司单指标跨年份趋势查询。"
        "仅用于已批准口径内的年度趋势，不接受季度、同比或环比。"
        "适用于用户询问某公司某项指标的历史变化或近几年趋势。"
        "示例：宁德时代近三年营业收入趋势"
    ),
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    EXACT_METRIC_LOOKUP: ("company", "metric", "year"),
    LATEST_METRIC_LOOKUP: ("company", "metric"),
    COMPARE_METRIC_LOOKUP: ("company", "metric"),
    TREND_METRIC_LOOKUP: ("company", "metric"),
}

EXAMPLE_QUESTIONS: dict[str, tuple[str, ...]] = {
    EXACT_METRIC_LOOKUP: ("宁德时代 2024 年营业收入是多少？", "腾讯 2024 年归母净利润是多少？"),
    LATEST_METRIC_LOOKUP: ("宁德时代最新营业收入是多少？", "腾讯最近归母净利润是多少？"),
    COMPARE_METRIC_LOOKUP: ("宁德时代和腾讯 2024 年营业收入对比", "宁德时代和腾讯 2024 年归母净利润对比"),
    TREND_METRIC_LOOKUP: ("宁德时代近三年营业收入趋势", "腾讯历年归母净利润趋势"),
}


def template_catalog_text() -> str:
    lines: list[str] = [APPROVED_CANONICAL_SCOPE_TEXT]
    for template_id in VALID_TEMPLATE_IDS:
        required = ", ".join(REQUIRED_FIELDS[template_id])
        examples = " / ".join(EXAMPLE_QUESTIONS[template_id])
        lines.append(
            f"- {template_id}: {QUERY_DESCRIPTIONS[template_id]}; "
            f"required={required}; examples={examples}"
        )
    return "\n".join(lines)


__all__ = [
    "COMPARE_METRIC_LOOKUP",
    "APPROVED_CANONICAL_SCOPE_TEXT",
    "EXACT_METRIC_LOOKUP",
    "EXAMPLE_QUESTIONS",
    "LATEST_METRIC_LOOKUP",
    "QUERY_DESCRIPTIONS",
    "REQUIRED_FIELDS",
    "TREND_METRIC_LOOKUP",
    "VALID_TEMPLATE_IDS",
    "template_catalog_text",
]
