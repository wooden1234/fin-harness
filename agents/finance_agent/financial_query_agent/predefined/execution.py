"""predefined 纯执行节点，对应 assistgen predefined_cypher/node.py。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.state import (
    PredefinedSqlInputState,
    PredefinedSqlOutputState,
)
from agents.finance_agent.financial_query_agent.predefined.quarter_aggregate import (
    aggregate_quarter_rows,
)
from agents.finance_agent.financial_query_agent.predefined.sql_builder import (
    build_sql_from_resolution,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist import (
    BuiltPredefinedSql,
    PredefinedTemplateRegistry,
    ResolvedPredefinedQuery,
)
from agents.finance_agent.financial_query_agent.services.fact_service import (
    FinancialFactService,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)


@dataclass(frozen=True)
class PredefinedExecutionResult:
    template_id: str
    sql: str
    params: dict[str, Any]
    rows: list[FinancialSqlResultRow]
    missing_fields: list[str]
    errors: list[str]


async def build_predefined_sql_query(
    template_id: str,
    intent: FinancialQueryIntent,
    *,
    limit: int = 5,
) -> BuiltPredefinedSql:
    """查表白名单并绑定 SQL 参数。"""
    return await PredefinedTemplateRegistry.build(template_id, intent, limit=limit)


def build_predefined_sql_from_resolution(
    resolved_query: ResolvedPredefinedQuery,
    *,
    limit: int = 5,
) -> BuiltPredefinedSql:
    """基于已解析的字典结果构建 SQL，不再二次做实体查询。"""
    return PredefinedTemplateRegistry.build_from_resolution(
        resolved_query,
        limit=limit,
    )


async def execute_predefined_sql(
    state: PredefinedSqlInputState,
    *,
    limit: int = 5,
) -> PredefinedSqlOutputState:
    """执行白名单 SQL：lookup dict → bind params → run query。"""
    task = state.get("task", "")
    params = state.get("query_parameters", {})
    template_id = str(params.get("template_id") or "").strip()
    intent = state.get("intent")
    resolved_query = state.get("resolved_query")
    errors: list[str] = []

    if not template_id or template_id not in PredefinedTemplateRegistry.valid_template_ids():
        errors.append(f"Unable to find the specified SQL template: {template_id}")
        return PredefinedSqlOutputState(
            task=task,
            template_id=template_id,
            statement="",
            parameters={},
            rows=[],
            missing_fields=["template"],
            errors=errors,
            steps=["execute_predefined_sql"],
        )

    if not isinstance(intent, FinancialQueryIntent):
        errors.append("Missing FinancialQueryIntent for predefined execution")
        return PredefinedSqlOutputState(
            task=task,
            template_id=template_id,
            statement="",
            parameters={},
            rows=[],
            missing_fields=["intent"],
            errors=errors,
            steps=["execute_predefined_sql"],
        )

    if isinstance(resolved_query, ResolvedPredefinedQuery):
        built = build_predefined_sql_from_resolution(
            resolved_query,
            limit=limit,
        )
    else:
        built = await build_predefined_sql_query(template_id, intent, limit=limit)
    if built.missing_fields:
        return PredefinedSqlOutputState(
            task=task,
            template_id=template_id,
            statement=built.sql,
            parameters=built.params,
            rows=[],
            missing_fields=built.missing_fields,
            errors=errors,
            steps=["execute_predefined_sql"],
        )

    rows = await FinancialFactService.run_generated_sql(
        built.sql,
        params=built.params,
        limit=limit,
    )
    if isinstance(resolved_query, ResolvedPredefinedQuery) and resolved_query.metric_bindings:
        rows = aggregate_quarter_rows(rows, resolved_query.metric_bindings)
    return PredefinedSqlOutputState(
        task=task,
        template_id=template_id,
        statement=built.sql,
        parameters=built.params,
        rows=rows,
        missing_fields=[],
        errors=errors,
        steps=["execute_predefined_sql"],
    )


async def execute_predefined_sql_query(
    template_id: str,
    intent: FinancialQueryIntent,
    *,
    limit: int = 5,
) -> PredefinedExecutionResult:
    """兼容旧接口：直接按 template_id + intent 执行。"""
    output = await execute_predefined_sql(
        {
            "task": "",
            "query_name": "predefined_sql",
            "query_parameters": {"template_id": template_id},
            "intent": intent,
            "steps": [],
        },
        limit=limit,
    )
    return PredefinedExecutionResult(
        template_id=output["template_id"],
        sql=output["statement"],
        params=output["parameters"],
        rows=output["rows"],
        missing_fields=output["missing_fields"],
        errors=output["errors"],
    )


__all__ = [
    "PredefinedExecutionResult",
    "build_predefined_sql_from_resolution",
    "build_predefined_sql_query",
    "execute_predefined_sql",
    "execute_predefined_sql_query",
]
