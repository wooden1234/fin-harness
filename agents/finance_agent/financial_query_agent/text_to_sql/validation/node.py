"""text_to_sql 规则门校验（安全 / 白名单 / 参数）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing_extensions import Literal

from agents.finance_agent.financial_query_agent.services.sql_executor import FinancialSqlExecutor, SqlValidationError

ValidationErrorType = Literal[
    "",
    "safety",
    "schema",
    "parameter",
    "semantic",
    "runtime",
    "result_empty",
    "result_schema",
]


@dataclass(frozen=True)
class SqlValidationResult:
    validated_sql: str
    error: str = ""
    error_type: ValidationErrorType = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _classify_validation_error(error: str) -> ValidationErrorType:
    if "绑定参数" in error or "未使用参数" in error:
        return "parameter"
    if "非白名单表" in error:
        return "schema"
    return "safety"


def validate_generated_sql(
    sql: str,
    *,
    params: dict[str, Any] | None = None,
) -> SqlValidationResult:
    """规则门：只读安全、白名单表、参数绑定；语义与结果合理性在后续节点校验。"""
    try:
        validated_sql = FinancialSqlExecutor.validate_readonly_sql(sql, params=params)
        return SqlValidationResult(validated_sql=validated_sql)
    except SqlValidationError as exc:
        error = str(exc)
        return SqlValidationResult(
            validated_sql="",
            error=error,
            error_type=_classify_validation_error(error),
        )


__all__ = ["SqlValidationResult", "ValidationErrorType", "validate_generated_sql"]
