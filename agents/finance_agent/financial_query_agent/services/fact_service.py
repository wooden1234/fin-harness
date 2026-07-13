"""annual_financial_facts 结构化查询服务（供 financial_query 使用）。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.models.annual_financial_fact import AnnualFinancialFact
from agents.finance_agent.financial_query_agent.services.fact_search_executor import FinancialFactSearchExecutor
from agents.finance_agent.financial_query_agent.services.query_router import FinancialQueryRouter, FinancialQueryTemplate
from agents.finance_agent.financial_query_agent.services.schemas import FinancialQueryIntent, FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.services.sql_executor import FinancialSqlExecutor
from agents.finance_agent.financial_query_agent.services.sql_templates import FinancialSqlTemplateRegistry
from agents.finance_agent.financial_query_agent.services.template_executor import FinancialTemplateExecutor


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
            display_company=cls._display_company,
            display_metric_name=cls._display_metric_name,
            fact_year=cls._fact_year,
            fact_year_sort_key_desc=lambda fact: cls._fact_year_sort_key(fact, desc=True),
            fact_year_sort_key_asc=lambda fact: cls._fact_year_sort_key(fact, desc=False),
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

    @staticmethod
    def _document(fact: AnnualFinancialFact):
        table = getattr(fact, "table", None)
        return getattr(table, "document", None) if table is not None else None

    @staticmethod
    def _table(fact: AnnualFinancialFact):
        return getattr(fact, "table", None)

    @staticmethod
    def _metric(fact: AnnualFinancialFact):
        return getattr(fact, "metric", None)

    @classmethod
    def _display_company(cls, fact: AnnualFinancialFact) -> str:
        document = cls._document(fact)
        company = getattr(document, "company", None) if document is not None else None
        if company is not None and getattr(company, "name", None):
            return company.name
        title = getattr(fact, "title", None) or getattr(document, "title", "") or ""
        if " Annual Report" in title:
            return title.split(" Annual Report")[0]
        ticker = getattr(fact, "ticker", None) or getattr(company, "ticker", None)
        return title or ticker or "未知公司"

    @classmethod
    def _display_metric_name(cls, fact: AnnualFinancialFact) -> str:
        metric = cls._metric(fact)
        return getattr(fact, "metric_name", None) or getattr(metric, "canonical_name", "未知指标")

    @classmethod
    def _fact_year(cls, fact: AnnualFinancialFact) -> int | str:
        document = cls._document(fact)
        return fact.period_year or getattr(fact, "fiscal_year", None) or getattr(document, "fiscal_year", None) or "未知年份"

    @classmethod
    def _fact_year_sort_key(cls, fact: AnnualFinancialFact, *, desc: bool) -> tuple[int, int]:
        year = cls._fact_year(fact)
        if isinstance(year, int):
            return (0, -year if desc else year)
        return (1, 0)

    @staticmethod
    def _display_value(fact: AnnualFinancialFact) -> str:
        if fact.raw_value:
            return fact.raw_value
        if fact.value is None:
            return "—"
        if isinstance(fact.value, Decimal):
            normalized = fact.value.normalize()
            return format(normalized, "f").rstrip("0").rstrip(".")
        return str(fact.value)

    @staticmethod
    def _display_sql_value(row: FinancialSqlResultRow) -> str:
        return row.raw_value or row.value or "—"

    @classmethod
    def format_answer(cls, facts: list[AnnualFinancialFact]) -> str:
        if not facts:
            return "（数据库中未找到匹配的财务指标，建议改查 PDF 文档库。）"
        lines: list[str] = []
        for fact in facts:
            company = cls._display_company(fact)
            document = cls._document(fact)
            year = cls._fact_year(fact)
            value = cls._display_value(fact)
            unit = fact.unit or ""
            currency = f"，{fact.currency}" if fact.currency else ""
            table = cls._table(fact)
            page_num = getattr(fact, "page_num", None) or getattr(table, "page_num", None)
            page = f"第{page_num}页" if page_num else "未知页码"
            metric_name = cls._display_metric_name(fact)
            source = getattr(fact, "source", None) or getattr(document, "source", "")
            lines.append(f"{company} {year}年 {metric_name}为 {value}{unit}{currency}（来源：{source} {page}）")
        return "\n".join(lines)

    @classmethod
    def format_sql_answer(
        cls,
        rows: list[FinancialSqlResultRow],
        *,
        include_source: bool = False,
    ) -> str:
        if not rows:
            return "（数据库中未找到匹配的财务指标，建议改查 PDF 文档库。）"
        lines: list[str] = []
        for row in rows:
            year = row.period_year or row.fiscal_year or "未知年份"
            value = cls._display_sql_value(row)
            currency = f"，{row.currency}" if row.currency else ""
            if include_source:
                page = f"第{row.page_num}页" if row.page_num else "未知页码"
                lines.append(
                    f"{row.company_name} {year}年 {row.metric_name}为 {value}{row.unit}{currency}"
                    f"（来源：{row.source} {page}）"
                )
            else:
                lines.append(
                    f"{row.company_name} {year}年 {row.metric_name}为 {value}{row.unit}{currency}"
                )
        return "\n".join(lines)

    @staticmethod
    def to_citations(facts: list[AnnualFinancialFact]) -> list[dict]:
        citations: list[dict] = []
        for fact in facts:
            document = FinancialFactService._document(fact)
            table = FinancialFactService._table(fact)
            metric_name = FinancialFactService._display_metric_name(fact)
            snippet = f"{metric_name}: {fact.raw_value or fact.value}"
            if fact.unit:
                snippet = f"{snippet}{fact.unit}"
            citation: dict = {"source": getattr(fact, "source", None) or getattr(document, "source", None) or getattr(fact, "doc_id", None) or getattr(document, "doc_id", "") or "", "snippet": snippet[:200]}
            page_num = getattr(fact, "page_num", None) or getattr(table, "page_num", None)
            if page_num is not None:
                citation["page"] = page_num
            citations.append(citation)
        return citations

    @staticmethod
    def sql_rows_to_citations(rows: list[FinancialSqlResultRow]) -> list[dict]:
        citations: list[dict] = []
        for row in rows:
            snippet = f"{row.metric_name}: {row.raw_value or row.value}"
            if row.unit:
                snippet = f"{snippet}{row.unit}"
            citation: dict[str, Any] = {"source": row.source or row.doc_id, "snippet": snippet[:200]}
            if row.page_num is not None:
                citation["page"] = row.page_num
            citations.append(citation)
        return citations


__all__ = ["FinancialFactService", "FinancialQueryTemplate"]
