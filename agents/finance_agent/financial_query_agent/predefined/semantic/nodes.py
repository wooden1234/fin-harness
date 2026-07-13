"""语义层 node 函数。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.canonical_metric_registry import (
    CanonicalMetricRegistry,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import (
    ResolvedCompany,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.coverage_resolver import (
    CoverageResolver,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CanonicalMetricMatch,
    CoverageRequest,
    CoverageResolution,
)


async def resolve_canonical_metrics_node(
    intent: FinancialQueryIntent,
    *,
    companies_by_canonical: dict[str, ResolvedCompany] | None = None,
) -> list[CanonicalMetricMatch]:
    return await CanonicalMetricRegistry.resolve(
        intent,
        companies_by_canonical=companies_by_canonical,
    )


async def resolve_coverage_node(
    intent: FinancialQueryIntent,
    canonical_matches: list[CanonicalMetricMatch],
    template_id: str,
) -> CoverageResolution:
    query_type = intent.operation
    return await CoverageResolver.resolve(
        CoverageRequest(
            canonical_matches=canonical_matches,
            companies=intent.companies,
            years=list(intent.years),
            query_type=query_type,
            template_id=template_id,
        )
    )


__all__ = ["resolve_canonical_metrics_node", "resolve_coverage_node"]
