"""coverage resolver：判断指标在当前公司/年份/粒度下是否可查，并选择口径策略。"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricMatch,
    CompanyCoverage,
    CoverageRequest,
    CoverageResolution,
    CoverageStrategy,
)
from app.core.database import AsyncSessionLocal
from app.models.annual_financial_fact import (
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
)

QUARTER_LABELS = ("第一季度", "第二季度", "第三季度", "第四季度")


class CoverageResolver:
    """在指标已标准化之后，判断当前查询口径是否可满足。"""

    @classmethod
    async def resolve(cls, request: CoverageRequest) -> CoverageResolution:
        if not request.canonical_matches:
            return CoverageResolution(
                status="unavailable",
                unavailable_reason="未识别到有效指标语义",
            )
        primary = request.canonical_matches[0]
        if not primary.canonical_metric_code:
            return CoverageResolution(
                status="unavailable",
                unavailable_reason=f"无法识别指标：{primary.requested_metric}",
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
            return CoverageResolution(
                status="unavailable",
                canonical_metric_code=primary.canonical_metric_code,
                unavailable_reason="未找到任何公司级指标映射",
            )
        status, answer_policy, clarify_reason = cls._aggregate_status(
            company_coverages,
            query_type=request.query_type,
        )
        return CoverageResolution(
            status=status,
            canonical_metric_code=primary.canonical_metric_code,
            company_coverages=company_coverages,
            answer_policy=answer_policy,
            clarify_reason=clarify_reason,
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
                    AnnualFinancialFact.period_label,
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
        quarter_labels_by_year: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            period_type = row.period_type or "annual"
            year = row.year
            if year is not None:
                availability[period_type].add(int(year))
            if period_type == "quarter" and year is not None and row.period_label:
                quarter_labels_by_year[int(year)].add(row.period_label)
        for year, labels in quarter_labels_by_year.items():
            if all(label in labels for label in QUARTER_LABELS):
                availability["quarter_complete"].add(year)
        return availability

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
        quarter_complete_years = availability.get("quarter_complete", set())
        all_years = sorted(annual_years | quarter_complete_years | availability.get("quarter", set()))
        available_period_types = sorted(
            key for key in availability if key != "quarter_complete" and availability[key]
        )
        target_year = years[0] if years else (max(all_years) if all_years else None)

        strategy: CoverageStrategy = "unavailable"
        selected_year = target_year

        if query_type == "latest":
            if annual_years:
                strategy = "annual_direct"
                selected_year = max(annual_years)
            elif all_years:
                strategy = "latest_available"
                selected_year = max(all_years)
        elif query_type == "lookup":
            if target_year is not None and target_year in annual_years:
                strategy = "annual_direct"
            elif target_year is not None and target_year in quarter_complete_years:
                strategy = "sum_quarters"
            elif annual_years:
                strategy = "annual_direct"
                selected_year = max(annual_years)
            elif quarter_complete_years:
                strategy = "sum_quarters"
                selected_year = max(quarter_complete_years)
        elif query_type == "compare":
            if target_year is not None and target_year in annual_years:
                strategy = "annual_direct"
            elif target_year is not None and target_year in quarter_complete_years:
                strategy = "sum_quarters"
            elif annual_years:
                strategy = "annual_direct"
                selected_year = max(annual_years)
            elif quarter_complete_years:
                strategy = "sum_quarters"
                selected_year = max(quarter_complete_years)
            else:
                strategy = "unavailable"
        elif query_type == "trend":
            if annual_years:
                strategy = "annual_direct"
            elif quarter_complete_years:
                strategy = "sum_quarters"
            elif all_years:
                strategy = "latest_available"

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
    ) -> tuple[str, str, str]:
        strategies = {item.selected_strategy for item in coverages}
        if all(item.selected_strategy == "unavailable" for item in coverages):
            return "unavailable", "unavailable", ""
        if query_type == "compare":
            usable = [item for item in coverages if item.selected_strategy != "unavailable"]
            if not usable:
                return "unavailable", "unavailable", ""
            if len(usable) < len(coverages):
                return "partial", "partial_compare", "部分公司缺少可比数据"
            if len(strategies - {"unavailable"}) > 1:
                return "partial", "compare_with_mixed_source_metrics", ""
            return "ok", "direct", ""
        if any(item.selected_strategy == "sum_quarters" for item in coverages):
            if all(item.selected_strategy in {"sum_quarters", "annual_direct"} for item in coverages):
                if len(strategies - {"unavailable"}) > 1:
                    return "partial", "sum_quarters_disclosure", ""
                return "ok", "sum_quarters_disclosure", ""
        if any(item.selected_strategy == "clarify_for_granularity" for item in coverages):
            return "clarify", "clarify_for_granularity", "存在多种可用口径，需确认查询粒度"
        return "ok", "direct", ""


__all__ = ["CoverageResolver"]
