"""coverage resolver：判断指标在当前公司/年份下是否有年报可查。

predefined 只承接年报口径；季度数据不在白名单路径内汇总，
若年报缺失仅有季度，则标记为 unavailable 并说明理由，由上游转 text_to_sql。
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    AnswerPolicy,
    CanonicalMetricMatch,
    CompanyCoverage,
    CoverageRequest,
    CoverageResolution,
    CoverageStatus,
    CoverageStrategy,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.reason_codes import (
    QUARTER_ONLY_FALLBACK_REASON,
    CoverageReasonCode,
    render_coverage_reasons,
)
from app.core.database import AsyncSessionLocal
from app.models.finance.annual_financial_fact import (
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
)


class CoverageResolver:
    """在指标已标准化之后，判断当前查询年报口径是否可满足。"""

    @classmethod
    def _resolution(
        cls,
        *,
        status: CoverageStatus,
        answer_policy: AnswerPolicy = "unavailable",
        reason_code: CoverageReasonCode | None = None,
        requested_metric: str = "",
        **kwargs,
    ) -> CoverageResolution:
        clarify_reason, unavailable_reason = render_coverage_reasons(
            reason_code,
            requested_metric=requested_metric,
        )
        return CoverageResolution(
            status=status,
            answer_policy=answer_policy,
            reason_code=reason_code,
            clarify_reason=clarify_reason,
            unavailable_reason=unavailable_reason,
            **kwargs,
        )

    @classmethod
    async def resolve(cls, request: CoverageRequest) -> CoverageResolution:
        if not request.canonical_matches:
            return cls._resolution(
                status="unavailable",
                reason_code="METRIC_SEMANTICS_MISSING",
            )
        primary = request.canonical_matches[0]
        if not primary.canonical_metric_code:
            return cls._resolution(
                status="unavailable",
                reason_code="METRIC_UNRECOGNIZED",
                requested_metric=primary.requested_metric,
            )
        company_coverages: list[CompanyCoverage] = []
        for match in primary.company_metric_matches:
            if match.metric_id is None or match.company_id is None:
                continue
            availability = await cls._probe_availability(
                company_id=match.company_id,
                metric_id=match.metric_id,
            )
            coverage = cls._select_strategy(
                company_key=match.company_key,
                company_id=match.company_id,
                metric_id=match.metric_id,
                canonical_metric_code=primary.canonical_metric_code,
                metric_name=match.metric_name,
                availability=availability,
                years=request.years,
                query_type=request.query_type,
            )
            company_coverages.append(coverage)
        if not company_coverages:
            return cls._resolution(
                status="unavailable",
                canonical_metric_code=primary.canonical_metric_code,
                reason_code="COMPANY_METRIC_MAPPING_MISSING",
            )
        status, answer_policy, reason_code = cls._aggregate_status(
            company_coverages,
            query_type=request.query_type,
        )
        return cls._resolution(
            status=status,
            canonical_metric_code=primary.canonical_metric_code,
            company_coverages=company_coverages,
            answer_policy=answer_policy,
            reason_code=reason_code,
            requested_metric=primary.requested_metric,
        )

    @classmethod
    async def _probe_availability(
        cls,
        *,
        company_id: int,
        metric_id: int,
    ) -> dict[str, set[int]]:
        async with AsyncSessionLocal() as session:
            stmt = (
                select(
                    AnnualFinancialFact.period_type,
                    func.coalesce(AnnualFinancialFact.period_year, AnnualReportDocument.fiscal_year).label(
                        "year"
                    ),
                )
                .join(AnnualFinancialTable, AnnualFinancialTable.id == AnnualFinancialFact.table_id)
                .join(AnnualReportDocument, AnnualReportDocument.id == AnnualFinancialTable.document_id)
                .where(
                    AnnualReportDocument.company_id == company_id,
                    AnnualFinancialFact.metric_id == metric_id,
                    AnnualFinancialFact.period_type.notin_(("change_rate", "unknown")),
                )
            )
            rows = (await session.execute(stmt)).all()
        availability: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            period_type = row.period_type or "annual"
            year = row.year
            if year is not None:
                availability[period_type].add(int(year))
        return availability

    @classmethod
    def _has_quarter_data(cls, availability: dict[str, set[int]], years: list[int] | None = None) -> bool:
        quarter_years = availability.get("quarter", set())
        if not quarter_years:
            return False
        if not years:
            return True
        return bool(quarter_years.intersection(years))

    @classmethod
    def _select_strategy(
        cls,
        *,
        company_key: str,
        company_id: int,
        metric_id: int,
        canonical_metric_code: str,
        metric_name: str,
        availability: dict[str, set[int]],
        years: list[int],
        query_type: str,
    ) -> CompanyCoverage:
        annual_years = availability.get("annual", set()) | availability.get("period_end", set())
        all_years = sorted(annual_years | availability.get("quarter", set()))
        available_period_types = sorted(key for key in availability if availability[key])
        target_year = years[0] if years else (max(annual_years) if annual_years else None)

        strategy: CoverageStrategy = "unavailable"
        selected_year = target_year

        if query_type == "latest":
            if annual_years:
                strategy = "annual_direct"
                selected_year = max(annual_years)
            elif cls._has_quarter_data(availability):
                strategy = "quarter_only"
                selected_year = None
            else:
                strategy = "unavailable"
                selected_year = None
        elif query_type == "lookup":
            if target_year is not None and target_year in annual_years:
                strategy = "annual_direct"
                selected_year = target_year
            elif target_year is not None and cls._has_quarter_data(availability, [target_year]):
                strategy = "quarter_only"
                selected_year = target_year
            else:
                strategy = "unavailable"
                selected_year = target_year
        elif query_type == "compare":
            if target_year is not None and target_year in annual_years:
                strategy = "annual_direct"
                selected_year = target_year
            elif target_year is not None and cls._has_quarter_data(availability, [target_year]):
                strategy = "quarter_only"
                selected_year = target_year
            else:
                strategy = "unavailable"
                selected_year = target_year
        elif query_type in {"compare_year", "trend"}:
            requested = set(years) if years else set()
            if requested:
                annual_hits = annual_years.intersection(requested)
                if annual_hits:
                    strategy = "annual_direct"
                    selected_year = None
                elif cls._has_quarter_data(availability, years):
                    strategy = "quarter_only"
                    selected_year = None
                else:
                    strategy = "unavailable"
                    selected_year = None
            elif annual_years:
                strategy = "annual_direct"
                selected_year = None
            elif cls._has_quarter_data(availability):
                strategy = "quarter_only"
                selected_year = None
            else:
                strategy = "unavailable"
                selected_year = None

        return CompanyCoverage(
            company_key=company_key,
            company_id=company_id,
            metric_id=metric_id,
            canonical_metric_code=canonical_metric_code,
            metric_name=metric_name,
            available_period_types=available_period_types,
            available_years=sorted(all_years),
            selected_strategy=strategy,
            selected_year=selected_year,
        )

    @classmethod
    def _aggregate_status(
        cls,
        coverages: list[CompanyCoverage],
        *,
        query_type: str,
    ) -> tuple[CoverageStatus, AnswerPolicy, CoverageReasonCode | None]:
        strategies = {item.selected_strategy for item in coverages}
        if "quarter_only" in strategies and strategies <= {"quarter_only", "unavailable"}:
            return "unavailable", "unavailable", "QUARTER_ONLY_NO_ANNUAL"
        if all(item.selected_strategy == "unavailable" for item in coverages):
            return "unavailable", "unavailable", "ANNUAL_DATA_NOT_FOUND"
        if query_type == "compare":
            usable = [item for item in coverages if item.selected_strategy == "annual_direct"]
            if not usable:
                if "quarter_only" in strategies:
                    return "unavailable", "unavailable", "QUARTER_ONLY_NO_ANNUAL"
                return "unavailable", "unavailable", "ANNUAL_DATA_NOT_FOUND"
            if len(usable) < len(coverages):
                # 部分公司仅有季度：整单不在 predefined 内拼凑，转 text_to_sql
                if any(item.selected_strategy == "quarter_only" for item in coverages):
                    return "unavailable", "unavailable", "QUARTER_ONLY_NO_ANNUAL"
                return "partial", "partial_compare", "PARTIAL_COMPARE_MISSING_ANNUAL"
            return "ok", "direct", None
        if any(item.selected_strategy == "clarify_for_granularity" for item in coverages):
            return "clarify", "clarify_for_granularity", "GRANULARITY_CLARIFY_NEEDED"
        if any(item.selected_strategy == "quarter_only" for item in coverages):
            return "unavailable", "unavailable", "QUARTER_ONLY_NO_ANNUAL"
        return "ok", "direct", None


__all__ = ["CoverageResolver", "QUARTER_ONLY_FALLBACK_REASON"]
