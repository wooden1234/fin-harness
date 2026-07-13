"""白名单 SQL 字典，对应 assistgen predefined_cypher/cypher_dict.py。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    COMPARE_METRIC_LOOKUP,
    COMPARE_YEAR_METRIC_LOOKUP,
    EXACT_METRIC_LOOKUP,
    LATEST_METRIC_LOOKUP,
    TREND_METRIC_LOOKUP,
    VALID_TEMPLATE_IDS,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.schema import (
    TEMPLATE_SELECT_SQL,
)


@dataclass(frozen=True)
class PredefinedSqlDefinition:
    template_id: str
    sql: str


_MULTI_YEAR_SQL = (
    TEMPLATE_SELECT_SQL
    + "\n  AND (:years_empty OR COALESCE(fact.period_year, document.fiscal_year) IN :years)"
    + "\nORDER BY company.name, metric.canonical_name, COALESCE(fact.period_year, document.fiscal_year) ASC"
    + "\nLIMIT :limit\n"
)

PREDEFINED_SQL_DICT: dict[str, PredefinedSqlDefinition] = {
    EXACT_METRIC_LOOKUP: PredefinedSqlDefinition(
        template_id=EXACT_METRIC_LOOKUP,
        sql=(
            TEMPLATE_SELECT_SQL
            + "\n  AND COALESCE(fact.period_year, document.fiscal_year) IN :years"
            + "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, metric.canonical_name"
            + "\nLIMIT :limit\n"
        ),
    ),
    LATEST_METRIC_LOOKUP: PredefinedSqlDefinition(
        template_id=LATEST_METRIC_LOOKUP,
        sql=(
            TEMPLATE_SELECT_SQL
            + "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, metric.canonical_name"
            + "\nLIMIT :limit\n"
        ),
    ),
    COMPARE_METRIC_LOOKUP: PredefinedSqlDefinition(
        template_id=COMPARE_METRIC_LOOKUP,
        sql=(
            TEMPLATE_SELECT_SQL
            + "\n  AND (:years_empty OR COALESCE(fact.period_year, document.fiscal_year) IN :years)"
            + "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, company.name, metric.canonical_name"
            + "\nLIMIT :limit\n"
        ),
    ),
    COMPARE_YEAR_METRIC_LOOKUP: PredefinedSqlDefinition(
        template_id=COMPARE_YEAR_METRIC_LOOKUP,
        sql=_MULTI_YEAR_SQL,
    ),
    TREND_METRIC_LOOKUP: PredefinedSqlDefinition(
        template_id=TREND_METRIC_LOOKUP,
        sql=_MULTI_YEAR_SQL,
    ),
}

assert set(PREDEFINED_SQL_DICT) == set(VALID_TEMPLATE_IDS)


__all__ = ["PREDEFINED_SQL_DICT", "PredefinedSqlDefinition"]
