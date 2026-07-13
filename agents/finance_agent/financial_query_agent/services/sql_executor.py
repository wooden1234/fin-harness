"""只读 SQL 安全执行器。"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import bindparam, text

from app.core.database import AsyncSessionLocal
from agents.finance_agent.financial_query_agent.predefined.whitelist.schema import (
    ALLOWED_TABLES,
)
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow

_FORBIDDEN_SQL_RE = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|comment|copy|call|do|merge)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z0-9_.]+)", re.IGNORECASE)
_NAMED_PARAM_RE = re.compile(r"(?<!:):([a-zA-Z_][a-zA-Z0-9_]*)")


class SqlValidationError(ValueError):
    """SQL 安全校验失败。"""


class FinancialSqlExecutor:
    """执行 financial_query 的只读 SQL。"""

    ALLOWED_TABLES = set(ALLOWED_TABLES)

    @classmethod
    def validate_readonly_sql(cls, sql: str, *, params: dict[str, Any] | None = None) -> str:
        normalized = sql.strip()
        if not normalized:
            raise SqlValidationError("SQL 为空")
        if normalized.endswith(";"):
            normalized = normalized[:-1].strip()
        if ";" in normalized:
            raise SqlValidationError("不允许多条 SQL 语句")
        lowered = normalized.lower()
        if not lowered.startswith("select"):
            raise SqlValidationError("仅允许执行 SELECT 语句")
        if _FORBIDDEN_SQL_RE.search(normalized):
            raise SqlValidationError("SQL 包含危险关键字")
        tables = {match.group(1).lower() for match in _TABLE_RE.finditer(normalized)}
        unknown_tables = {table for table in tables if table not in cls.ALLOWED_TABLES}
        if unknown_tables:
            raise SqlValidationError(f"SQL 使用了非白名单表: {', '.join(sorted(unknown_tables))}")
        if not re.search(r"\blimit\b", lowered):
            normalized = f"{normalized}\nLIMIT :__system_limit"
        if params is not None:
            cls.validate_sql_parameters(normalized, params)
        return normalized

    @staticmethod
    def extract_named_parameters(sql: str) -> set[str]:
        return set(_NAMED_PARAM_RE.findall(sql))

    @classmethod
    def validate_sql_parameters(cls, sql: str, params: dict[str, Any]) -> None:
        expected = cls.extract_named_parameters(sql)
        provided = set(params)
        missing = expected - provided - {"__system_limit"}
        extra = provided - expected - {"__system_limit"}
        if missing:
            raise SqlValidationError(f"SQL 缺少绑定参数: {', '.join(sorted(missing))}")
        if extra:
            raise SqlValidationError(f"SQL 包含未使用参数: {', '.join(sorted(extra))}")

    @classmethod
    async def execute(cls, sql: str, *, params: dict[str, Any] | None = None, limit: int = 5) -> list[FinancialSqlResultRow]:
        validated_sql = cls.validate_readonly_sql(sql, params=params)
        bound_params = dict(params or {})
        bound_params.setdefault("__system_limit", max(1, limit))
        stmt = text(validated_sql)
        for key, value in bound_params.items():
            if isinstance(value, list):
                stmt = stmt.bindparams(bindparam(key, expanding=True))
        async with AsyncSessionLocal() as session:
            result = await session.execute(stmt, bound_params)
            rows = result.mappings().all()
        return [cls._to_result_row(row) for row in rows]

    @staticmethod
    def _to_result_row(row: Any) -> FinancialSqlResultRow:
        mapping = dict(row)
        return FinancialSqlResultRow(
            company_id=mapping.get("company_id"),
            company_name=str(mapping.get("company_name") or "未知公司"),
            ticker=str(mapping.get("ticker") or ""),
            fiscal_year=mapping.get("fiscal_year"),
            period_year=mapping.get("period_year"),
            period_label=str(mapping.get("period_label") or ""),
            period_type=str(mapping.get("period_type") or ""),
            metric_name=str(mapping.get("metric_name") or "未知指标"),
            raw_value=str(mapping.get("raw_value") or ""),
            value=str(mapping.get("value") or ""),
            unit=str(mapping.get("unit") or ""),
            currency=str(mapping.get("currency") or ""),
            source=str(mapping.get("source") or ""),
            page_num=mapping.get("page_num"),
            doc_id=str(mapping.get("doc_id") or ""),
        )


__all__ = ["FinancialSqlExecutor", "SqlValidationError"]
