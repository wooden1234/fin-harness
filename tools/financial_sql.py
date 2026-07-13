"""财务 SQL 工具，包装现有 financial_query_agent services。"""

from __future__ import annotations

from typing import Any

from agents.finance_agent.financial_query_agent.services.fact_service import (
    FinancialFactService,
)
from agents.finance_agent.financial_query_agent.services.sql_executor import (
    FinancialSqlExecutor,
)


async def validate_sql(sql: str) -> str:
    return FinancialSqlExecutor.validate_select_sql(sql)


async def execute_sql(
    sql: str,
    *,
    params: dict[str, Any] | None = None,
    limit: int = 5,
):
    return await FinancialFactService.run_generated_sql(
        sql,
        params=params or {},
        limit=limit,
    )
