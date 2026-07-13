"""公司实体解析：桥接 EntityResolver canonical key 与 DB financial_companies。"""

from __future__ import annotations

from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from agents.finance_agent.financial_query_agent.services.entity_resolver import (
    EntityResolver,
)
from app.core.database import AsyncSessionLocal
from app.models.annual_financial_fact import FinancialCompany


class ResolvedCompany(BaseModel):
    """统一的公司解析结果。"""

    canonical_key: str = Field(description="语义层 canonical 名，如 Tencent、CATL")
    company_id: int
    db_company_key: str = Field(description="financial_companies.company_key，如 TCEHY")
    name: str
    ticker: str | None = None


class CompanyResolver:
    """将用户公司表达解析为 canonical key + company_id。"""

    @classmethod
    async def resolve(cls, companies: list[str]) -> list[ResolvedCompany]:
        if not companies:
            return []

        exact_terms: set[str] = set()
        ticker_terms: set[str] = set()
        requested_canonicals: set[str] = set()
        for company in companies:
            cleaned = company.strip()
            if not cleaned:
                continue
            canonical = EntityResolver._canonical_company(cleaned)
            if canonical:
                requested_canonicals.add(canonical)
            for term in EntityResolver.expand_company_terms(cleaned):
                term_cleaned = term.strip()
                if not term_cleaned:
                    continue
                if term_cleaned.isdigit():
                    ticker_terms.add(term_cleaned)
                else:
                    exact_terms.add(term_cleaned.lower())

        if not exact_terms and not ticker_terms:
            return []

        async with AsyncSessionLocal() as session:
            conditions = []
            if exact_terms:
                conditions.extend(
                    [
                        func.lower(FinancialCompany.name).in_(exact_terms),
                        func.lower(FinancialCompany.company_key).in_(exact_terms),
                    ]
                )
            if ticker_terms:
                conditions.append(FinancialCompany.ticker.in_(ticker_terms))
            stmt = select(
                FinancialCompany.id,
                FinancialCompany.company_key,
                FinancialCompany.name,
                FinancialCompany.ticker,
            ).where(or_(*conditions))
            rows = (await session.execute(stmt)).all()

        resolved_by_id: dict[int, ResolvedCompany] = {}
        for row in rows:
            canonical_key = cls._canonical_from_db_row(
                name=row.name,
                company_key=row.company_key,
                ticker=row.ticker,
            )
            resolved_by_id[row.id] = ResolvedCompany(
                canonical_key=canonical_key,
                company_id=row.id,
                db_company_key=row.company_key,
                name=row.name,
                ticker=row.ticker,
            )

        if not requested_canonicals:
            return list(resolved_by_id.values())

        return [
            item
            for item in resolved_by_id.values()
            if item.canonical_key in requested_canonicals
        ] or list(resolved_by_id.values())

    @classmethod
    async def resolve_by_canonical(cls, companies: list[str]) -> dict[str, ResolvedCompany]:
        return {item.canonical_key: item for item in await cls.resolve(companies)}

    @classmethod
    async def resolve_company_ids(cls, companies: list[str]) -> list[int]:
        return list(dict.fromkeys(item.company_id for item in await cls.resolve(companies)))

    @classmethod
    def _canonical_from_db_row(
        cls,
        *,
        name: str,
        company_key: str,
        ticker: str | None,
    ) -> str:
        for candidate in (company_key, name, ticker or ""):
            canonical = EntityResolver._match_exact(candidate, EntityResolver.COMPANY_ALIASES)
            if canonical is not None:
                return canonical
        return company_key


__all__ = ["CompanyResolver", "ResolvedCompany"]
