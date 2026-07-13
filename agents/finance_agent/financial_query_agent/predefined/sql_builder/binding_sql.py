"""公司级 metric binding SQL 构造。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    ResolvedMetricBinding,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.schema import (
    FIN_CORE_SCHEMA,
    STANDARD_JOIN_SQL,
)


def build_period_filters(bindings: list[ResolvedMetricBinding]) -> str:
    """按 binding 生成年报 period_type 过滤子句（predefined 仅年报）。"""
    if not bindings:
        return ""
    clauses: list[str] = []
    for index, binding in enumerate(bindings):
        period_clause = (
            "(fact.period_type = 'annual' OR fact.period_type IS NULL) "
            "AND (fact.period_type IS NULL OR fact.period_type NOT IN ('change_rate', 'unknown'))"
        )
        clauses.append(
            f"(document.company_id = :company_id_{index} "
            f"AND fact.metric_id = :metric_id_{index} AND {period_clause})"
        )
    return "(\n    " + "\n    OR ".join(clauses) + "\n  )"


def build_metric_binding_where(bindings: list[ResolvedMetricBinding]) -> tuple[str, dict[str, int]]:
    """生成公司级 metric binding OR 子句及参数。"""
    if not bindings:
        return "1 = 0", {}
    clauses: list[str] = []
    params: dict[str, int] = {}
    for index, binding in enumerate(bindings):
        clauses.append(
            f"(document.company_id = :company_id_{index} AND fact.metric_id = :metric_id_{index})"
        )
        params[f"company_id_{index}"] = binding.company_id
        params[f"metric_id_{index}"] = binding.metric_id
    return "(\n    " + "\n    OR ".join(clauses) + "\n  )", params


def build_binding_select_sql(bindings: list[ResolvedMetricBinding]) -> tuple[str, dict[str, int]]:
    """构建带公司级 binding 的基础 SELECT SQL。"""
    binding_where, params = build_metric_binding_where(bindings)
    period_where = build_period_filters(bindings)
    sql = f"""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  document.fiscal_year AS fiscal_year,
  fact.period_year AS period_year,
  fact.period_label AS period_label,
  fact.period_type AS period_type,
  metric.canonical_name AS metric_name,
  COALESCE(fact.raw_value, '') AS raw_value,
  COALESCE(CAST(fact.value AS TEXT), '') AS value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(fact.currency, '') AS currency,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id
{STANDARD_JOIN_SQL}
WHERE {binding_where}
  AND {period_where}
  AND fact.period_label IS NOT NULL
  AND fact.period_label != ''
  AND fact.period_label NOT LIKE 'value_%'
""".strip()
    return sql, params


def append_template_suffix(
    base_sql: str,
    template_id: str,
    *,
    years: list[int],
) -> str:
    """按模板追加年份排序与 LIMIT :limit。"""
    from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
        COMPARE_METRIC_LOOKUP,
        COMPARE_YEAR_METRIC_LOOKUP,
        EXACT_METRIC_LOOKUP,
        LATEST_METRIC_LOOKUP,
        TREND_METRIC_LOOKUP,
    )

    years_empty = not years
    year_values = years or [-1]
    suffix = ""
    if template_id == EXACT_METRIC_LOOKUP:
        suffix = (
            f"\n  AND COALESCE(fact.period_year, document.fiscal_year) IN ({', '.join(str(y) for y in year_values)})"
            f"\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, metric.canonical_name"
        )
    elif template_id == LATEST_METRIC_LOOKUP:
        suffix = (
            "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, metric.canonical_name"
        )
    elif template_id == COMPARE_METRIC_LOOKUP:
        if not years_empty:
            suffix = (
                f"\n  AND COALESCE(fact.period_year, document.fiscal_year) IN ({', '.join(str(y) for y in year_values)})"
                "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, company.name, metric.canonical_name"
            )
        else:
            suffix = "\nORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC, company.name, metric.canonical_name"
    elif template_id in {COMPARE_YEAR_METRIC_LOOKUP, TREND_METRIC_LOOKUP}:
        if not years_empty:
            suffix = (
                f"\n  AND COALESCE(fact.period_year, document.fiscal_year) IN ({', '.join(str(y) for y in year_values)})"
                "\nORDER BY company.name, metric.canonical_name, COALESCE(fact.period_year, document.fiscal_year) ASC"
            )
        else:
            suffix = (
                "\nORDER BY company.name, metric.canonical_name, COALESCE(fact.period_year, document.fiscal_year) ASC"
            )
    return f"{base_sql}{suffix}\nLIMIT :limit\n"


__all__ = [
    "append_template_suffix",
    "build_binding_select_sql",
    "build_metric_binding_where",
    "build_period_filters",
]
