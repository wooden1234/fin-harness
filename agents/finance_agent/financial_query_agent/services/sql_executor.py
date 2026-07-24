"""只读 SQL 安全执行器。"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from agents.finance_agent.financial_query_agent.predefined.whitelist.schema import (
    ALLOWED_TABLES,
)
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow

_NAMED_PARAM_RE = re.compile(r"(?<!:):([a-zA-Z_][a-zA-Z0-9_]*)")
_MAX_RESULT_ROWS = 100
_DANGEROUS_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_read_file",
        "pg_read_binary_file",
        "dblink",
        "lo_import",
        "lo_export",
        "nextval",
        "set_config",
    }
)
_MUTATING_NODE_NAMES = frozenset(
    {
        "Alter",
        "Command",
        "Create",
        "Delete",
        "Drop",
        "Insert",
        "Merge",
        "Set",
        "Truncate",
        "Update",
    }
)

# 结构化查询使用独立连接；生产环境应配置为只读数据库账号。
_financial_query_engine = create_async_engine(
    settings.FINANCIAL_QUERY_DATABASE_URL or settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
FinancialQuerySessionLocal = sessionmaker(
    bind=_financial_query_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


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
        try:
            statements = sqlglot.parse(normalized, read="postgres")
        except sqlglot.errors.ParseError as exc:
            raise SqlValidationError("SQL 语法错误") from exc
        if len(statements) != 1:
            raise SqlValidationError("不允许多条 SQL 语句")

        tree = statements[0]
        if not isinstance(tree, exp.Select):
            raise SqlValidationError("仅允许执行 SELECT 语句")
        if tree.find(exp.Into) or tree.args.get("locks"):
            raise SqlValidationError("禁止 SELECT INTO 或锁语句")

        for node in tree.walk():
            if type(node).__name__ in _MUTATING_NODE_NAMES:
                raise SqlValidationError("SQL 包含数据修改或 DDL 操作")
            if isinstance(node, exp.Anonymous):
                function_name = node.name.lower()
                if function_name in _DANGEROUS_FUNCTIONS:
                    raise SqlValidationError(f"禁止调用危险函数: {function_name}")

        cte_names = {
            cte.alias_or_name.lower()
            for cte in tree.find_all(exp.CTE)
            if cte.alias_or_name
        }
        unknown_tables: set[str] = set()
        for table in tree.find_all(exp.Table):
            table_name = table.name.lower()
            if table_name in cte_names:
                continue
            database = table.db.lower()
            qualified_name = f"{database}.{table_name}" if database else table_name
            if qualified_name not in cls.ALLOWED_TABLES:
                unknown_tables.add(qualified_name)
        if unknown_tables:
            raise SqlValidationError(f"SQL 使用了非白名单表: {', '.join(sorted(unknown_tables))}")

        limit_node = tree.args.get("limit")
        if limit_node is not None:
            limit_expression = limit_node.args.get("expression")
            if isinstance(limit_expression, exp.Literal) and not limit_expression.is_string:
                try:
                    limit_value = int(limit_expression.this)
                except ValueError as exc:
                    raise SqlValidationError("LIMIT 必须是整数或命名参数") from exc
                if not 1 <= limit_value <= _MAX_RESULT_ROWS:
                    raise SqlValidationError(f"LIMIT 必须在 1 到 {_MAX_RESULT_ROWS} 之间")
            elif isinstance(limit_expression, exp.Placeholder):
                parameter_name = limit_expression.name
                if params is not None and parameter_name in params:
                    try:
                        limit_value = int(params[parameter_name])
                    except (TypeError, ValueError) as exc:
                        raise SqlValidationError("LIMIT 参数必须是整数") from exc
                    if not 1 <= limit_value <= _MAX_RESULT_ROWS:
                        raise SqlValidationError(f"LIMIT 必须在 1 到 {_MAX_RESULT_ROWS} 之间")
            else:
                raise SqlValidationError("LIMIT 必须是整数或命名参数")
        else:
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
        bounded_limit = min(max(1, limit), _MAX_RESULT_ROWS)
        bound_params.setdefault("__system_limit", bounded_limit)
        try:
            parsed = sqlglot.parse_one(validated_sql, read="postgres")
            limit_node = parsed.args.get("limit")
            limit_expression = limit_node.args.get("expression") if limit_node else None
            if isinstance(limit_expression, exp.Placeholder):
                parameter_name = limit_expression.name
                if parameter_name in bound_params:
                    bound_params[parameter_name] = min(
                        int(bound_params[parameter_name]),
                        _MAX_RESULT_ROWS,
                    )
        except (ValueError, TypeError) as exc:
            raise SqlValidationError("LIMIT 参数必须是整数") from exc
        stmt = text(validated_sql)
        for key, value in bound_params.items():
            if isinstance(value, list):
                stmt = stmt.bindparams(bindparam(key, expanding=True))
        timeout_ms = max(1, int(settings.FINANCIAL_QUERY_STATEMENT_TIMEOUT_MS))
        async with FinancialQuerySessionLocal() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'"))
            result = await session.execute(stmt, bound_params)
            rows = result.mappings().all()
            await session.rollback()
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
            canonical_code=str(mapping.get("canonical_code") or ""),
            raw_value=str(mapping.get("raw_value") or ""),
            value=str(mapping.get("value") or ""),
            unit=str(mapping.get("unit") or ""),
            currency=str(mapping.get("currency") or ""),
            source=str(mapping.get("source") or ""),
            page_num=mapping.get("page_num"),
            doc_id=str(mapping.get("doc_id") or ""),
            document_id=mapping.get("document_id"),
            table_id=mapping.get("table_id"),
            source_cell_id=mapping.get("source_cell_id"),
            section=str(mapping.get("section") or ""),
        )


__all__ = ["FinancialSqlExecutor", "SqlValidationError"]
