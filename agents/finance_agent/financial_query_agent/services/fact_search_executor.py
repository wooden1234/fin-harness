"""金融结构化事实召回执行器。"""

from __future__ import annotations

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import joinedload

from app.core.database import AsyncSessionLocal
from app.models.annual_financial_fact import (
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
    FinancialCompany,
    FinancialMetric,
)
from agents.finance_agent.financial_query_agent.services.entity_resolver import EntityResolver
from agents.finance_agent.financial_query_agent.predefined.intent import FinancialQueryIntent


class FinancialFactSearchExecutor:
    """执行底层结构化事实召回。"""

    @staticmethod
    def resolve_company_terms(company: str) -> list[str]:
        return EntityResolver.expand_company_terms(company)

    @staticmethod
    def resolve_metric_terms(metric: str, *, company: str | None = None) -> list[str]:
        return EntityResolver.expand_metric_terms(metric, company=company)

    @classmethod
    def resolve_metric_terms_for_query(cls, query: FinancialQueryIntent) -> list[str]:
        return EntityResolver.expand_metric_terms_for_companies(query.metrics, query.companies)

    @classmethod
    async def search(
        cls,
        query: FinancialQueryIntent,
        *,
        limit: int = 5,
        latest_only: bool = False,
    ) -> list[AnnualFinancialFact]:
        company_terms: list[str] = []
        for company in query.companies:
            company_terms.extend(cls.resolve_company_terms(company))
        metric_terms: list[str] = cls.resolve_metric_terms_for_query(query)
        if not company_terms and not query.company.strip().isdigit():
            return []
        async with AsyncSessionLocal() as session:
            conditions: list = []
            company_conditions = []
            for term in company_terms:
                company_conditions.append(FinancialCompany.name.ilike(f"%{term}%"))
                company_conditions.append(AnnualReportDocument.title.ilike(f"%{term}%"))
            for company in query.companies:
                ticker = company.strip()
                if ticker.isdigit():
                    company_conditions.append(FinancialCompany.ticker == ticker)
            if not company_conditions:
                return []
            conditions.append(or_(*company_conditions))
            if query.years:
                year_conditions = []
                for year in query.years:
                    year_conditions.append(AnnualFinancialFact.period_year == year)
                    year_conditions.append(and_(AnnualFinancialFact.period_year.is_(None), AnnualReportDocument.fiscal_year == year))
                conditions.append(or_(*year_conditions))
            if metric_terms:
                metric_conditions = []
                for term in metric_terms:
                    metric_conditions.append(FinancialMetric.canonical_name.ilike(f"%{term}%"))
                    metric_conditions.append(FinancialMetric.aliases.ilike(f"%{term}%"))
                conditions.append(or_(*metric_conditions))
            conditions.extend(cls.clean_row_filters())
            stmt = (
                select(AnnualFinancialFact)
                .join(AnnualFinancialFact.table)
                .join(AnnualFinancialTable.document)
                .outerjoin(AnnualReportDocument.company)
                .join(AnnualFinancialFact.metric)
                .options(
                    joinedload(AnnualFinancialFact.table).joinedload(AnnualFinancialTable.document).joinedload(AnnualReportDocument.company),
                    joinedload(AnnualFinancialFact.metric),
                )
                .where(and_(*conditions))
                .order_by(AnnualFinancialFact.period_year.desc(), AnnualReportDocument.fiscal_year.desc(), FinancialMetric.canonical_name)
                .limit(limit)
            )
            result = await session.execute(stmt)
            facts = list(result.scalars().all())
            if latest_only and facts:
                return facts[:1]
            return facts

    @staticmethod
    def clean_row_filters() -> list:
        return [
            AnnualFinancialFact.period_label.isnot(None),
            AnnualFinancialFact.period_label != "",
            ~AnnualFinancialFact.period_label.like("value_%"),
            or_(AnnualFinancialFact.period_type == "annual", AnnualFinancialFact.period_type.is_(None)),
            or_(AnnualFinancialFact.period_type.is_(None), ~AnnualFinancialFact.period_type.in_(["change_rate", "unknown"])),
        ]


__all__ = ["FinancialFactSearchExecutor"]
