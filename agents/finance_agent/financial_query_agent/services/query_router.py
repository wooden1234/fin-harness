"""金融查询内部路由策略。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.finance_agent.financial_query_agent.predefined.intent import FinancialQueryIntent


@dataclass(frozen=True)
class FinancialQueryTemplate:
    """预定义查询模板；未命中时再回退到通用搜索。"""

    name: str
    description: str


class FinancialQueryRouter:
    """决定金融查询应走模板、低风险通用搜索还是后续兜底策略。"""

    SAFE_GENERIC_SEARCH_ROUTE = "generic_search_safe"
    NEEDS_CLARIFICATION_ROUTE = "needs_clarification"
    TEXT_TO_SQL_FALLBACK_ROUTE = "text_to_sql_fallback"

    EXACT_LOOKUP_TEMPLATE = FinancialQueryTemplate(
        name="exact_metric_lookup",
        description="恰好 1 公司 + 恰好 1 年份 + 恰好 1 指标精确查数",
    )
    LATEST_LOOKUP_TEMPLATE = FinancialQueryTemplate(
        name="latest_metric_lookup",
        description="恰好 1 公司 + 恰好 1 指标，查询最新已发布年度",
    )
    COMPARE_LOOKUP_TEMPLATE = FinancialQueryTemplate(
        name="compare_metric_lookup",
        description="≥2 公司 + 恰好 1 显式共同年份 + 恰好 1 指标横向对比",
    )
    COMPARE_YEAR_LOOKUP_TEMPLATE = FinancialQueryTemplate(
        name="compare_year_metric_lookup",
        description="恰好 1 公司 + ≥2 显式年份 + 恰好 1 指标跨年对比",
    )
    TREND_LOOKUP_TEMPLATE = FinancialQueryTemplate(
        name="trend_metric_lookup",
        description="恰好 1 公司 + 恰好 1 指标 + ≥2 个显式年份的年度趋势",
    )

    @classmethod
    def match_template(cls, question: str, query: FinancialQueryIntent) -> FinancialQueryTemplate | None:
        normalized_question = question.strip()
        has_company = len(query.companies) == 1 and bool(query.company.strip())
        has_metric = len(query.metrics) == 1 and bool(query.metric.strip())
        has_year = len(query.years) == 1 and query.year is not None
        has_multiple_companies = len(query.companies) > 1
        has_multiple_years = len(query.years) > 1
        has_multiple_metrics = len(query.metrics) > 1
        compare_keywords = ("对比", "比较", "对照")
        is_compare_phrasing = any(keyword in normalized_question for keyword in compare_keywords)
        if query.has_template_blocking_ambiguity():
            return None
        if (
            query.operation == "compare"
            and has_multiple_companies
            and has_metric
            and has_year
            and not has_multiple_metrics
            and not has_multiple_years
        ):
            return cls.COMPARE_LOOKUP_TEMPLATE
        if (
            query.operation in {"compare_year", "compare"}
            and has_company
            and has_metric
            and has_multiple_years
            and not has_multiple_metrics
            and (query.operation == "compare_year" or is_compare_phrasing)
        ):
            return cls.COMPARE_YEAR_LOOKUP_TEMPLATE
        if query.operation == "trend" and has_company and has_metric and has_multiple_years:
            return cls.TREND_LOOKUP_TEMPLATE
        if query.operation == "latest" and has_company and has_metric and not query.years:
            return cls.LATEST_LOOKUP_TEMPLATE
        if has_company and has_metric and has_year and not has_multiple_metrics:
            return cls.EXACT_LOOKUP_TEMPLATE
        if (
            has_multiple_companies
            and has_metric
            and has_year
            and not has_multiple_metrics
            and not has_multiple_years
        ):
            return cls.COMPARE_LOOKUP_TEMPLATE
        if has_company and has_metric and has_multiple_years and is_compare_phrasing:
            return cls.COMPARE_YEAR_LOOKUP_TEMPLATE
        if has_company and has_metric and has_multiple_years:
            return cls.TREND_LOOKUP_TEMPLATE
        if (
            has_company
            and has_metric
            and not query.years
            and any(keyword in normalized_question for keyword in ["最新", "最近", "当前"])
        ):
            return cls.LATEST_LOOKUP_TEMPLATE
        return None

    @classmethod
    def route_generic_search(cls, query: FinancialQueryIntent) -> str:
        if query.has_template_blocking_ambiguity():
            return cls.NEEDS_CLARIFICATION_ROUTE
        if cls.is_low_risk_generic_search(query):
            return cls.SAFE_GENERIC_SEARCH_ROUTE
        return cls.TEXT_TO_SQL_FALLBACK_ROUTE

    @staticmethod
    def is_low_risk_generic_search(query: FinancialQueryIntent) -> bool:
        return query.operation == "lookup" and len(query.companies) == 1 and len(query.metrics) >= 1


__all__ = ["FinancialQueryRouter", "FinancialQueryTemplate"]
