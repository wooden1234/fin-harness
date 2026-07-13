"""白名单 SQL 构建与实体 ID 解析。"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, or_, select

from app.core.database import AsyncSessionLocal
from app.models.annual_financial_fact import FinancialCompany, FinancialMetric
from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    REQUIRED_FIELDS,
    VALID_TEMPLATE_IDS,
    template_catalog_text,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.sql_dict import PREDEFINED_SQL_DICT
from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import (
    CompanyResolver,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CoverageResolution,
    ResolvedMetricBinding,
)
from agents.finance_agent.financial_query_agent.services.entity_resolver import (
    EntityResolver,
)

@dataclass(frozen=True)
class BuiltPredefinedSql:
    template_id: str
    sql: str
    params: dict[str, object]
    missing_fields: list[str]


@dataclass(frozen=True)
class ResolvedPredefinedQuery:
    """字典解析结果：模板已选定，实体已标准化并绑定到主键。"""

    template_id: str
    intent: FinancialQueryIntent
    company_ids: list[int]
    metric_bindings: list[ResolvedMetricBinding] = field(default_factory=list)
    coverage: CoverageResolution | None = None
    missing_fields: list[str] = field(default_factory=list)

    @property
    def metric_ids(self) -> list[int]:
        return list(dict.fromkeys(binding.metric_id for binding in self.metric_bindings))


class PredefinedTemplateRegistry:
    """查表白名单并绑定参数，对应 assistgen 执行前的 dict lookup + param bind。"""

    @classmethod
    def valid_template_ids(cls) -> set[str]:
        return set(VALID_TEMPLATE_IDS)

    @classmethod
    def template_examples(cls) -> str:
        return template_catalog_text()

    @classmethod
    async def resolve_intent(
        cls,
        template_id: str,
        intent: FinancialQueryIntent,
    ) -> ResolvedPredefinedQuery:
        """基于标准化意图做实体 ID 解析。"""
        if template_id not in PREDEFINED_SQL_DICT:
            return ResolvedPredefinedQuery(
                template_id=template_id,
                intent=intent,
                company_ids=[],
                metric_bindings=[],
                missing_fields=["template"],
            )

        required_fields = REQUIRED_FIELDS[template_id]
        company_ids = await cls._resolve_company_ids(intent.companies)
        metric_ids = await cls._resolve_metric_ids(intent.metrics, intent.companies)
        metric_bindings: list[ResolvedMetricBinding] = []
        target_company_ids = company_ids or [0]
        for company_id in target_company_ids:
            for metric_id in metric_ids:
                metric_bindings.append(
                    ResolvedMetricBinding(
                        company_id=company_id,
                        metric_id=metric_id,
                        canonical_metric_code="",
                        selected_strategy="annual_direct",
                    )
                )
        missing_fields = cls._missing_fields(
            required_fields,
            query=intent,
            company_ids=company_ids,
            metric_ids=metric_ids,
            years=list(intent.years),
        )
        return ResolvedPredefinedQuery(
            template_id=template_id,
            intent=intent,
            company_ids=company_ids,
            metric_bindings=metric_bindings,
            missing_fields=missing_fields,
        )

    @classmethod
    def build_from_resolution(
        cls,
        resolved_query: ResolvedPredefinedQuery,
        *,
        limit: int = 5,
    ) -> BuiltPredefinedSql:
        """执行阶段只根据已解析结果绑定参数，不再查询字典。"""
        template_id = resolved_query.template_id
        if template_id not in PREDEFINED_SQL_DICT:
            return BuiltPredefinedSql(
                template_id=template_id,
                sql="",
                params={},
                missing_fields=["template"],
            )
        if resolved_query.missing_fields:
            return BuiltPredefinedSql(
                template_id=template_id,
                sql="",
                params={},
                missing_fields=list(resolved_query.missing_fields),
            )

        from agents.finance_agent.financial_query_agent.predefined.sql_builder import (
            build_sql_from_resolution,
        )

        return build_sql_from_resolution(
            resolved_query,
            resolved_query.coverage,
            limit=limit,
        )

    @classmethod
    async def build(
        cls,
        template_id: str,
        query: FinancialQueryIntent,
        *,
        limit: int = 5,
    ) -> BuiltPredefinedSql:
        resolved_query = await cls.resolve_intent(template_id, query)
        return cls.build_from_resolution(resolved_query, limit=limit)

    @staticmethod
    def _missing_fields(
        required_fields: tuple[str, ...],
        *,
        query: FinancialQueryIntent,
        company_ids: list[int],
        metric_ids: list[int],
        years: list[int],
    ) -> list[str]:
        missing_fields: list[str] = []
        if "company" in required_fields and (not query.companies or not company_ids):
            missing_fields.append("company")
        if "metric" in required_fields and (not query.metrics or not metric_ids):
            missing_fields.append("metric")
        if "year" in required_fields and not years:
            missing_fields.append("year")
        return missing_fields

    @staticmethod
    async def _resolve_company_ids(companies: list[str]) -> list[int]:
        return await CompanyResolver.resolve_company_ids(companies)

    @staticmethod
    async def _resolve_metric_ids(metrics: list[str], companies: list[str] | None = None) -> list[int]:
        exact_terms: set[str] = set()
        fuzzy_terms: set[str] = set()
        for metric in metrics:
            for term in EntityResolver.expand_metric_terms(
                metric,
                company=companies[0] if companies else None,
            ):
                cleaned = term.strip()
                if not cleaned:
                    continue
                exact_terms.add(cleaned.lower())
                fuzzy_terms.add(cleaned.lower())
        if not exact_terms and not fuzzy_terms:
            return []
        async with AsyncSessionLocal() as session:
            conditions = []
            if exact_terms:
                conditions.append(func.lower(FinancialMetric.canonical_name).in_(exact_terms))
            for term in fuzzy_terms:
                conditions.append(func.lower(FinancialMetric.aliases).like(f"%{term}%"))
            stmt = select(FinancialMetric.id).where(or_(*conditions))
            result = await session.execute(stmt)
            return list(dict.fromkeys(result.scalars().all()))


__all__ = [
    "BuiltPredefinedSql",
    "PredefinedTemplateRegistry",
    "ResolvedPredefinedQuery",
]
