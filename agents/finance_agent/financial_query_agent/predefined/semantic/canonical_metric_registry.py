"""canonical metric registry：用户指标语义 -> 统一 canonical -> 公司级 source metric。"""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import (
    CompanyResolver,
    ResolvedCompany,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricMatch,
    CompanyMetricMatch,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.registry_seed import (
    CANONICAL_METRICS,
    COMPANY_OVERRIDES,
    company_metric_names,
    resolve_canonical_code,
)
from agents.finance_agent.financial_query_agent.services.entity_resolver import (
    EntityResolver,
)
from agents.finance_agent.financial_query_agent.services.errors import classify_exception
from app.core.database import AsyncSessionLocal
from app.models.finance.annual_financial_fact import (
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
    CanonicalMetric,
    CanonicalMetricAlias,
    CompanyMetricMapping,
    FinancialMetric,
)


class CanonicalMetricRegistry:
    """将用户说法标准化成统一财务语义，再映射到公司级可查询指标。"""

    @classmethod
    async def resolve(
        cls,
        intent: FinancialQueryIntent,
        *,
        companies_by_canonical: dict[str, ResolvedCompany] | None = None,
    ) -> list[CanonicalMetricMatch]:
        companies = intent.companies or [""]
        companies_by_canonical = companies_by_canonical or await CompanyResolver.resolve_by_canonical(
            intent.companies
        )
        matches: list[CanonicalMetricMatch] = []
        for requested_metric in intent.metrics:
            db_canonical = await cls._resolve_canonical_from_db(requested_metric)
            canonical_code = (
                db_canonical[0]
                if db_canonical is not None
                else resolve_canonical_code(requested_metric)
            )
            if not canonical_code:
                matches.append(
                    CanonicalMetricMatch(
                        canonical_metric_code="",
                        canonical_metric_name="",
                        requested_metric=requested_metric,
                        company_metric_matches=[],
                    )
                )
                continue
            canonical_name = (
                db_canonical[1]
                if db_canonical is not None
                else CANONICAL_METRICS[canonical_code].name
            )
            company_matches: list[CompanyMetricMatch] = []
            for company in companies:
                company_key = EntityResolver._canonical_company(company) if company else ""
                resolved_company = companies_by_canonical.get(company_key)
                mapped = await cls._resolve_company_mapping(
                    company_key=company_key,
                    company_id=resolved_company.company_id if resolved_company else None,
                    canonical_code=canonical_code,
                    target_year=cls._primary_year(intent.years),
                )
                if mapped is not None:
                    company_matches.append(mapped)
                    continue

                metric_names = company_metric_names(company_key, canonical_code)
                if not metric_names:
                    metric_names = [canonical_name]
                has_override = bool(
                    company_key
                    and canonical_code in COMPANY_OVERRIDES.get(company_key, {})
                )
                resolved = await cls._resolve_company_metric(
                    company_key=company_key,
                    company_id=resolved_company.company_id if resolved_company else None,
                    metric_names=metric_names,
                    has_override=has_override,
                )
                if resolved is not None:
                    company_matches.append(resolved)
            matches.append(
                CanonicalMetricMatch(
                    canonical_metric_code=canonical_code,
                    canonical_metric_name=canonical_name,
                    requested_metric=requested_metric,
                    company_metric_matches=company_matches,
                )
            )
        return matches

    @staticmethod
    def _normalize_alias(value: str) -> str:
        return value.replace(" ", "").lower()

    @staticmethod
    def _primary_year(years: list[int]) -> int | None:
        return years[0] if years else None

    @classmethod
    async def _resolve_canonical_from_db(cls, metric_text: str) -> tuple[str, str] | None:
        normalized = cls._normalize_alias(metric_text.strip())
        if not normalized:
            return None

        try:
            async with AsyncSessionLocal() as session:
                alias_stmt = (
                    select(CanonicalMetric.code, CanonicalMetric.name)
                    .join(CanonicalMetricAlias, CanonicalMetricAlias.canonical_code == CanonicalMetric.code)
                    .where(
                        CanonicalMetricAlias.normalized_alias == normalized,
                        CanonicalMetricAlias.is_active.is_(True),
                        CanonicalMetric.is_active.is_(True),
                    )
                    .order_by(CanonicalMetricAlias.priority.asc())
                    .limit(1)
                )
                row = (await session.execute(alias_stmt)).first()
                if row is not None:
                    return row.code, row.name

                metric_stmt = (
                    select(CanonicalMetric.code, CanonicalMetric.name)
                    .where(
                        func.lower(CanonicalMetric.code) == normalized,
                        CanonicalMetric.is_active.is_(True),
                    )
                    .limit(1)
                )
                row = (await session.execute(metric_stmt)).first()
                if row is not None:
                    return row.code, row.name

                name_stmt = (
                    select(CanonicalMetric.code, CanonicalMetric.name)
                    .where(
                        func.lower(CanonicalMetric.name) == normalized,
                        CanonicalMetric.is_active.is_(True),
                    )
                    .limit(1)
                )
                row = (await session.execute(name_stmt)).first()
                return (row.code, row.name) if row is not None else None
        except SQLAlchemyError as exc:
            # 数据库异常不能伪装成“未命中”，交由上层按基础设施故障降级并告警。
            failure = classify_exception(exc, source="canonical_metric_registry")
            raise RuntimeError(failure.code) from exc

    @classmethod
    async def _resolve_company_mapping(
        cls,
        *,
        company_key: str,
        company_id: int | None,
        canonical_code: str,
        target_year: int | None,
    ) -> CompanyMetricMatch | None:
        if company_id is None:
            return None

        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(
                        CompanyMetricMapping.source_metric_id,
                        CompanyMetricMapping.source_metric_name,
                        CompanyMetricMapping.confidence,
                    )
                    .where(
                        CompanyMetricMapping.company_id == company_id,
                        CompanyMetricMapping.canonical_code == canonical_code,
                        CompanyMetricMapping.is_active.is_(True),
                        CompanyMetricMapping.review_status == "approved",
                    )
                    .order_by(CompanyMetricMapping.priority.asc(), CompanyMetricMapping.id.asc())
                )
                if target_year is not None:
                    stmt = stmt.where(
                        or_(
                            CompanyMetricMapping.valid_from_year.is_(None),
                            CompanyMetricMapping.valid_from_year <= target_year,
                        ),
                        or_(
                            CompanyMetricMapping.valid_to_year.is_(None),
                            CompanyMetricMapping.valid_to_year >= target_year,
                        ),
                    )
                row = (await session.execute(stmt.limit(1))).first()
        except SQLAlchemyError as exc:
            # 语义治理表异常不能静默退回旧覆盖规则。
            failure = classify_exception(exc, source="company_metric_mapping")
            raise RuntimeError(failure.code) from exc

        if row is None:
            return None
        return CompanyMetricMatch(
            company_key=company_key,
            company_id=company_id,
            metric_id=row.source_metric_id,
            metric_name=row.source_metric_name,
            match_type="company_override",
            confidence=float(row.confidence or 0.0),
        )

    @classmethod
    async def _resolve_company_metric(
        cls,
        *,
        company_key: str,
        company_id: int | None,
        metric_names: list[str],
        has_override: bool,
    ) -> CompanyMetricMatch | None:
        if not metric_names:
            return None
        metric_id, metric_name = await cls._lookup_metric_id(
            company_id=company_id,
            metric_names=metric_names,
        )
        if metric_id is None:
            return CompanyMetricMatch(
                company_key=company_key,
                company_id=company_id,
                metric_id=None,
                metric_name=metric_names[0],
                match_type="company_override" if has_override else "global_alias",
                confidence=0.0,
            )
        return CompanyMetricMatch(
            company_key=company_key,
            company_id=company_id,
            metric_id=metric_id,
            metric_name=metric_name,
            match_type="company_override" if has_override else "global_alias",
            confidence=0.98 if has_override else 0.95,
        )

    @classmethod
    async def _lookup_metric_id(
        cls,
        *,
        company_id: int | None,
        metric_names: list[str],
    ) -> tuple[int | None, str]:
        lowered_names = [name.lower() for name in metric_names if name.strip()]
        if not lowered_names:
            return None, ""

        async with AsyncSessionLocal() as session:
            if company_id is not None:
                exact_id, exact_name = await cls._lookup_metric_for_company(
                    session,
                    company_id=company_id,
                    lowered_names=lowered_names,
                    metric_names=metric_names,
                    exact_only=True,
                )
                if exact_id is not None:
                    return exact_id, exact_name

                fuzzy_id, fuzzy_name = await cls._lookup_metric_for_company(
                    session,
                    company_id=company_id,
                    lowered_names=lowered_names,
                    metric_names=metric_names,
                    exact_only=False,
                )
                if fuzzy_id is not None:
                    return fuzzy_id, fuzzy_name
                return None, metric_names[0]

            stmt = (
                select(FinancialMetric.id, FinancialMetric.canonical_name)
                .where(func.lower(FinancialMetric.canonical_name).in_(lowered_names))
                .limit(1)
            )
            row = (await session.execute(stmt)).first()
            if row is not None:
                return row.id, row.canonical_name
        return None, metric_names[0]

    @staticmethod
    async def _lookup_metric_for_company(
        session,
        *,
        company_id: int,
        lowered_names: list[str],
        metric_names: list[str],
        exact_only: bool,
    ) -> tuple[int | None, str | None]:
        base = (
            select(FinancialMetric.id, FinancialMetric.canonical_name)
            .join(AnnualFinancialFact, AnnualFinancialFact.metric_id == FinancialMetric.id)
            .join(AnnualFinancialTable, AnnualFinancialTable.id == AnnualFinancialFact.table_id)
            .join(AnnualReportDocument, AnnualReportDocument.id == AnnualFinancialTable.document_id)
            .where(AnnualReportDocument.company_id == company_id)
        )
        if exact_only:
            stmt = base.where(func.lower(FinancialMetric.canonical_name).in_(lowered_names)).limit(1)
            row = (await session.execute(stmt)).first()
            return (row.id, row.canonical_name) if row is not None else (None, None)

        for name in metric_names:
            stmt = base.where(
                func.lower(FinancialMetric.canonical_name).like(f"%{name.lower()}%")
            ).limit(1)
            row = (await session.execute(stmt)).first()
            if row is not None:
                return row.id, row.canonical_name
        return None, None


__all__ = ["CanonicalMetricRegistry"]
