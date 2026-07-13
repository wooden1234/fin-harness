"""predefined 槽位标准化。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.models import (
    PredefinedSlotExtraction,
)
from agents.finance_agent.financial_query_agent.services.entity_resolver import (
    EntityResolver,
)


def build_predefined_query_intent(slots: PredefinedSlotExtraction) -> FinancialQueryIntent:
    """将白名单槽位标准化为模板执行可消费的意图对象。"""
    companies, company_ambiguity = EntityResolver.resolve_companies(slots.companies)
    metrics, metric_ambiguity = EntityResolver.resolve_metrics(slots.metrics)
    time_scope = "unspecified"
    if slots.operation == "latest":
        time_scope = "latest"
    elif len(slots.years) == 1:
        time_scope = "single"
    elif len(slots.years) > 1:
        time_scope = "range"

    return FinancialQueryIntent(
        companies=companies,
        years=list(slots.years),
        metrics=metrics,
        operation=slots.operation,
        time_scope=time_scope,
        top_k=slots.top_k,
        ambiguity=[*company_ambiguity, *metric_ambiguity],
    )


__all__ = ["build_predefined_query_intent"]
