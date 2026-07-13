"""SQL builder node：根据 coverage 结果生成公司级 binding SQL。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CoverageResolution,
)
from agents.finance_agent.financial_query_agent.predefined.sql_builder.binding_sql import (
    append_template_suffix,
    build_binding_select_sql,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    LATEST_METRIC_LOOKUP,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.registry import (
    BuiltPredefinedSql,
    ResolvedPredefinedQuery,
)


def build_sql_from_resolution(
    resolved_query: ResolvedPredefinedQuery,
    coverage: CoverageResolution | None = None,
    *,
    limit: int = 5,
) -> BuiltPredefinedSql:
    """根据已解析结果与 coverage 构建 SQL。"""
    template_id = resolved_query.template_id
    if resolved_query.missing_fields:
        return BuiltPredefinedSql(
            template_id=template_id,
            sql="",
            params={},
            missing_fields=list(resolved_query.missing_fields),
        )
    if not resolved_query.metric_bindings:
        return BuiltPredefinedSql(
            template_id=template_id,
            sql="",
            params={},
            missing_fields=["metric"],
        )

    effective_limit = max(1, min(limit, max(resolved_query.intent.top_k, 1)))
    if template_id == LATEST_METRIC_LOOKUP:
        effective_limit = 1

    base_sql, binding_params = build_binding_select_sql(resolved_query.metric_bindings)
    years = list(resolved_query.intent.years)
    if coverage and coverage.company_coverages:
        selected_years = [
            item.selected_year
            for item in coverage.company_coverages
            if item.selected_year is not None
        ]
        if selected_years and not years:
            years = [max(selected_years)]
    sql = append_template_suffix(
        base_sql,
        template_id,
        years=years,
    )
    params: dict[str, object] = dict(binding_params)
    params["limit"] = effective_limit
    return BuiltPredefinedSql(
        template_id=template_id,
        sql=sql,
        params=params,
        missing_fields=[],
    )


__all__ = ["build_sql_from_resolution"]
