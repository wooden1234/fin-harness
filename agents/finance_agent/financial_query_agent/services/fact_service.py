"""annual_financial_facts 结构化查询服务（供 financial_query 使用）。"""

from __future__ import annotations

from typing import Any

from app.models.finance.annual_financial_fact import AnnualFinancialFact
from app.shared import Citation
from agents.finance_agent.financial_query_agent.services.citation_builder import (
    FinancialCitationBuilder,
)
from agents.finance_agent.financial_query_agent.services.fact_search_executor import FinancialFactSearchExecutor
from agents.finance_agent.financial_query_agent.services.query_router import FinancialQueryRouter, FinancialQueryTemplate
from agents.finance_agent.financial_query_agent.services.schemas import FinancialQueryIntent, FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.services.sql_executor import FinancialSqlExecutor
from agents.finance_agent.financial_query_agent.services.sql_templates import FinancialSqlTemplateRegistry
from agents.finance_agent.financial_query_agent.services.template_executor import FinancialTemplateExecutor
from agents.finance_agent.financial_query_agent.services.result_formatter import FinancialResultFormatter


class FinancialFactService:
    SAFE_GENERIC_SEARCH_ROUTE = FinancialQueryRouter.SAFE_GENERIC_SEARCH_ROUTE
    NEEDS_CLARIFICATION_ROUTE = FinancialQueryRouter.NEEDS_CLARIFICATION_ROUTE
    TEXT_TO_SQL_FALLBACK_ROUTE = FinancialQueryRouter.TEXT_TO_SQL_FALLBACK_ROUTE
    EXACT_LOOKUP_TEMPLATE = FinancialQueryRouter.EXACT_LOOKUP_TEMPLATE
    LATEST_LOOKUP_TEMPLATE = FinancialQueryRouter.LATEST_LOOKUP_TEMPLATE
    COMPARE_LOOKUP_TEMPLATE = FinancialQueryRouter.COMPARE_LOOKUP_TEMPLATE
    TREND_LOOKUP_TEMPLATE = FinancialQueryRouter.TREND_LOOKUP_TEMPLATE

    @staticmethod
    def resolve_company_terms(company: str) -> list[str]:
        return FinancialFactSearchExecutor.resolve_company_terms(company)

    @staticmethod
    def resolve_metric_terms(metric: str, *, company: str | None = None) -> list[str]:
        return FinancialFactSearchExecutor.resolve_metric_terms(metric, company=company)

    @classmethod
    def match_template(cls, question: str, query: FinancialQueryIntent) -> FinancialQueryTemplate | None:
        return FinancialQueryRouter.match_template(question, query)

    @classmethod
    async def execute_query(cls, question: str, query: FinancialQueryIntent, *, limit: int | None = None) -> tuple[list[AnnualFinancialFact], str]:
        limit = limit or query.top_k
        template = FinancialQueryRouter.match_template(question, query)
        if template is not None:
            facts = await cls.run_template(template, query, limit=limit)
            return facts, template.name
        generic_route = FinancialQueryRouter.route_generic_search(query)
        if generic_route == cls.SAFE_GENERIC_SEARCH_ROUTE:
            facts = await cls.search(query, limit=limit)
            return facts, generic_route
        return [], generic_route

    @classmethod
    def _route_generic_search(cls, query: FinancialQueryIntent) -> str:
        return FinancialQueryRouter.route_generic_search(query)

    @staticmethod
    def _is_low_risk_generic_search(query: FinancialQueryIntent) -> bool:
        return FinancialQueryRouter.is_low_risk_generic_search(query)

    @classmethod
    async def run_template(cls, template: FinancialQueryTemplate, query: FinancialQueryIntent, *, limit: int = 5) -> list[AnnualFinancialFact]:
        return await FinancialTemplateExecutor.run_template(
            template,
            query,
            search_fn=FinancialFactSearchExecutor.search,
            display_company=FinancialResultFormatter.display_company,
            display_metric_name=FinancialResultFormatter.display_metric_name,
            fact_year=FinancialResultFormatter.fact_year,
            fact_year_sort_key_desc=lambda fact: FinancialResultFormatter.fact_year_sort_key(fact, desc=True),
            fact_year_sort_key_asc=lambda fact: FinancialResultFormatter.fact_year_sort_key(fact, desc=False),
            limit=limit,
        )

    @classmethod
    async def search(cls, query: FinancialQueryIntent, *, limit: int = 5) -> list[AnnualFinancialFact]:
        return await FinancialFactSearchExecutor.search(query, limit=limit)

    @classmethod
    async def run_sql_template(cls, template_id: str, query: FinancialQueryIntent, *, limit: int = 5) -> tuple[list[FinancialSqlResultRow], str, dict[str, Any], list[str]]:
        built = await FinancialSqlTemplateRegistry.build(template_id, query, limit=limit)
        if built.missing_fields:
            return [], "", {}, built.missing_fields
        rows = await FinancialSqlExecutor.execute(built.sql, params=built.params, limit=limit)
        return rows, built.sql, built.params, []

    @classmethod
    async def run_generated_sql(cls, sql: str, *, params: dict[str, Any] | None = None, limit: int = 5) -> list[FinancialSqlResultRow]:
        return await FinancialSqlExecutor.execute(sql, params=params, limit=limit)

    @classmethod
    async def _search_base(cls, query: FinancialQueryIntent, *, limit: int = 5, latest_only: bool = False) -> list[AnnualFinancialFact]:
        return await FinancialFactSearchExecutor.search(query, limit=limit, latest_only=latest_only)

    @staticmethod
    def _clean_row_filters() -> list:
        return FinancialFactSearchExecutor.clean_row_filters()

    @classmethod
    def format_answer(cls, facts: list[AnnualFinancialFact]) -> str:
        return FinancialResultFormatter.format_answer(facts)

    @classmethod
    def format_sql_answer(cls, rows: list[FinancialSqlResultRow], *, include_source: bool = False) -> str:
        return FinancialResultFormatter.format_sql_answer(rows, include_source=include_source)

    @staticmethod
    def to_citations(facts: list[AnnualFinancialFact]) -> list[dict[str, Any]]:
        return FinancialCitationBuilder.to_citations(facts)

    @staticmethod
    def sql_rows_to_citations(
        rows: list[FinancialSqlResultRow],
        *,
        sub_task_id: str = "",
    ) -> list[Citation]:
        return FinancialCitationBuilder.sql_rows_to_citations(rows, sub_task_id=sub_task_id)


__all__ = ["FinancialFactService", "FinancialQueryTemplate"]
