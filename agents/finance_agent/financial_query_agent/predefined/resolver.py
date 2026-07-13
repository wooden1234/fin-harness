"""将语义层结果组装为 predefined 可执行解析结果。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricMatch,
    CoverageResolution,
    ResolvedMetricBinding,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    VALID_TEMPLATE_IDS,
    collect_slot_missing_fields,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import (
    CompanyResolver,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.registry import (
    ResolvedPredefinedQuery,
)


def build_metric_bindings(
    canonical_matches: list[CanonicalMetricMatch],
    coverage: CoverageResolution,
) -> list[ResolvedMetricBinding]:
    bindings: list[ResolvedMetricBinding] = []
    coverage_by_company = {item.company_id: item for item in coverage.company_coverages}
    for match in canonical_matches:
        for company_match in match.company_metric_matches:
            if company_match.metric_id is None or company_match.company_id is None:
                continue
            company_coverage = coverage_by_company.get(company_match.company_id)
            if company_coverage is None or company_coverage.selected_strategy == "unavailable":
                continue
            bindings.append(
                ResolvedMetricBinding(
                    company_id=company_match.company_id,
                    company_key=company_match.company_key,
                    metric_id=company_match.metric_id,
                    canonical_metric_code=match.canonical_metric_code,
                    metric_name=company_match.metric_name,
                    selected_strategy=company_coverage.selected_strategy,
                    selected_year=company_coverage.selected_year,
                )
            )
    return bindings


async def build_resolved_query(
    template_id: str,
    intent: FinancialQueryIntent,
    canonical_matches: list[CanonicalMetricMatch],
    coverage: CoverageResolution,
) -> ResolvedPredefinedQuery:
    if template_id not in VALID_TEMPLATE_IDS:
        return ResolvedPredefinedQuery(
            template_id=template_id,
            intent=intent,
            company_ids=[],
            metric_bindings=[],
            coverage=coverage,
            missing_fields=["template"],
        )

    company_ids = await CompanyResolver.resolve_company_ids(intent.companies)
    metric_bindings = build_metric_bindings(canonical_matches, coverage)
    missing_fields = _missing_fields(
        template_id,
        query=intent,
        company_ids=company_ids,
        metric_bindings=metric_bindings,
        years=list(intent.years),
        coverage=coverage,
    )
    return ResolvedPredefinedQuery(
        template_id=template_id,
        intent=intent,
        company_ids=company_ids,
        metric_bindings=metric_bindings,
        coverage=coverage,
        missing_fields=missing_fields,
    )


def _missing_fields(
    template_id: str,
    *,
    query: FinancialQueryIntent,
    company_ids: list[int],
    metric_bindings: list[ResolvedMetricBinding],
    years: list[int],
    coverage: CoverageResolution,
) -> list[str]:
    missing_fields = collect_slot_missing_fields(
        template_id,
        company_count=len(query.companies),
        resolved_company_count=len(company_ids),
        metric_count=len(query.metrics),
        has_resolved_metric=bool(metric_bindings),
        years=years,
    )
    if coverage.status == "unavailable":
        missing_fields.append("coverage")
    if coverage.status == "clarify":
        missing_fields.append("clarify")
    return missing_fields


__all__ = [
    "build_metric_bindings",
    "build_resolved_query",
]
