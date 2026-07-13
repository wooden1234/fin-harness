"""白名单 SQL 模板描述，供 tool_selection prompt 与 catalog 使用。"""

from __future__ import annotations

EXACT_METRIC_LOOKUP = "exact_metric_lookup"
LATEST_METRIC_LOOKUP = "latest_metric_lookup"
COMPARE_METRIC_LOOKUP = "compare_metric_lookup"
COMPARE_YEAR_METRIC_LOOKUP = "compare_year_metric_lookup"
TREND_METRIC_LOOKUP = "trend_metric_lookup"

# 当前 predefined 仅允许的已批准 canonical 口径。
APPROVED_CANONICAL_SCOPE_TEXT = (
    "当前白名单仅覆盖已批准口径：营业收入(REVENUE)、"
    "归属于上市公司股东的净利润(NET_INCOME_ATTR_PARENT)、"
    "营业利润(OPERATING_PROFIT)、"
    "经营活动产生的现金流量净额(OPERATING_CASHFLOW_NET)、"
    "研发费用(RND_EXPENSE)、"
    "毛利率(GROSS_MARGIN)、"
    "总资产(TOTAL_ASSETS)、"
    "总负债(TOTAL_LIABILITIES)、"
    "基本每股收益(EPS_BASIC)。"
    " 其他指标不要选 predefined，直接走 text_to_sql。"
)

VALID_TEMPLATE_IDS: frozenset[str] = frozenset(
    {
        EXACT_METRIC_LOOKUP,
        LATEST_METRIC_LOOKUP,
        COMPARE_METRIC_LOOKUP,
        COMPARE_YEAR_METRIC_LOOKUP,
        TREND_METRIC_LOOKUP,
    }
)

# 执行契约：只描述当前 SQL/校验真正能保证的能力，禁止夸大。
QUERY_DESCRIPTIONS: dict[str, str] = {
    EXACT_METRIC_LOOKUP: (
        "精确查数：恰好 1 家公司 + 恰好 1 个年份 + 恰好 1 个已批准指标。"
        "只查该年年报事实（period_type=annual），不做同比/环比/计算，也不做四季汇总。"
        "缺年报仅有季度时不可用，应转 text_to_sql。"
        "示例：宁德时代 2024 年营业收入是多少？"
    ),
    LATEST_METRIC_LOOKUP: (
        "最新年度查数：恰好 1 家公司 + 恰好 1 个已批准指标，且用户未指定年份。"
        "只返回最新已发布年报年度的一行；不是季度/半年度「最新一期」。"
        "仅有季度、无年报时不可用，应转 text_to_sql。"
        "示例：腾讯最新归母净利润是多少？"
    ),
    COMPARE_METRIC_LOOKUP: (
        "多公司横向对比：至少 2 家公司 + 恰好 1 个已批准指标 + 恰好 1 个显式共同年份。"
        "必须同年年报对齐；不支持各自最新年、跨年配对、多指标对比或四季汇总。"
        "示例：宁德时代和腾讯 2024 年营业收入对比"
    ),
    COMPARE_YEAR_METRIC_LOOKUP: (
        "单公司跨年对比：恰好 1 家公司 + 恰好 1 个已批准指标 + 至少 2 个显式年份。"
        "years 必须是具体年份列表；只取各年年报事实，不做同比增速计算。"
        "示例：宁德时代 2023 和 2024 年营业收入对比"
    ),
    TREND_METRIC_LOOKUP: (
        "年度趋势：恰好 1 家公司 + 恰好 1 个已批准指标 + 至少 2 个显式年份。"
        "years 必须是具体年份列表（如 [2022,2023,2024]）；只取年报事实。"
        "「近三年/近几年/历年」未展开成具体年份时不可用。"
        "示例：宁德时代 2022、2023、2024 年营业收入"
    ),
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    EXACT_METRIC_LOOKUP: ("company", "metric", "year"),
    LATEST_METRIC_LOOKUP: ("company", "metric"),
    COMPARE_METRIC_LOOKUP: ("company", "metric", "year"),
    COMPARE_YEAR_METRIC_LOOKUP: ("company", "metric", "year"),
    TREND_METRIC_LOOKUP: ("company", "metric", "year"),
}

EXAMPLE_QUESTIONS: dict[str, tuple[str, ...]] = {
    EXACT_METRIC_LOOKUP: ("宁德时代 2024 年营业收入是多少？", "腾讯 2024 年归母净利润是多少？"),
    LATEST_METRIC_LOOKUP: ("宁德时代最新营业收入是多少？", "腾讯最近归母净利润是多少？"),
    COMPARE_METRIC_LOOKUP: ("宁德时代和腾讯 2024 年营业收入对比", "宁德时代和腾讯 2024 年归母净利润对比"),
    COMPARE_YEAR_METRIC_LOOKUP: (
        "宁德时代 2023 和 2024 年营业收入对比",
        "腾讯 2022、2023 年归母净利润对比",
    ),
    TREND_METRIC_LOOKUP: (
        "宁德时代 2022、2023、2024 年营业收入",
        "腾讯 2021、2022、2023、2024 年归母净利润",
    ),
}


def collect_slot_missing_fields(
    template_id: str,
    *,
    company_count: int,
    resolved_company_count: int,
    metric_count: int,
    has_resolved_metric: bool,
    years: list[int],
) -> list[str]:
    """按模板执行契约检查槽位；返回缺失或不满足数量约束的字段名。"""
    missing: list[str] = []
    if template_id not in VALID_TEMPLATE_IDS:
        return ["template"]

    if template_id == EXACT_METRIC_LOOKUP:
        if company_count != 1 or resolved_company_count != 1:
            missing.append("company")
        if metric_count != 1 or not has_resolved_metric:
            missing.append("metric")
        if len(years) != 1:
            missing.append("year")
    elif template_id == LATEST_METRIC_LOOKUP:
        if company_count != 1 or resolved_company_count != 1:
            missing.append("company")
        if metric_count != 1 or not has_resolved_metric:
            missing.append("metric")
    elif template_id == COMPARE_METRIC_LOOKUP:
        if company_count < 2 or resolved_company_count < 2:
            missing.append("company")
        if metric_count != 1 or not has_resolved_metric:
            missing.append("metric")
        if len(years) != 1:
            missing.append("year")
    elif template_id in {COMPARE_YEAR_METRIC_LOOKUP, TREND_METRIC_LOOKUP}:
        if company_count != 1 or resolved_company_count != 1:
            missing.append("company")
        if metric_count != 1 or not has_resolved_metric:
            missing.append("metric")
        if len(years) < 2:
            missing.append("year")
    return missing


def template_catalog_text() -> str:
    lines: list[str] = [APPROVED_CANONICAL_SCOPE_TEXT]
    for template_id in (
        EXACT_METRIC_LOOKUP,
        LATEST_METRIC_LOOKUP,
        COMPARE_METRIC_LOOKUP,
        COMPARE_YEAR_METRIC_LOOKUP,
        TREND_METRIC_LOOKUP,
    ):
        required = ", ".join(REQUIRED_FIELDS[template_id])
        examples = " / ".join(EXAMPLE_QUESTIONS[template_id])
        lines.append(
            f"- {template_id}: {QUERY_DESCRIPTIONS[template_id]}; "
            f"required={required}; examples={examples}"
        )
    return "\n".join(lines)


__all__ = [
    "APPROVED_CANONICAL_SCOPE_TEXT",
    "COMPARE_METRIC_LOOKUP",
    "COMPARE_YEAR_METRIC_LOOKUP",
    "EXACT_METRIC_LOOKUP",
    "EXAMPLE_QUESTIONS",
    "LATEST_METRIC_LOOKUP",
    "QUERY_DESCRIPTIONS",
    "REQUIRED_FIELDS",
    "TREND_METRIC_LOOKUP",
    "VALID_TEMPLATE_IDS",
    "collect_slot_missing_fields",
    "template_catalog_text",
]
