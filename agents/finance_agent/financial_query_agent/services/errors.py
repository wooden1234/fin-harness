"""financial_query 内部故障分类，供状态、日志和告警复用。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError

FailureCategory = Literal[
    "user_clarification",
    "unsupported",
    "sql_unsafe",
    "no_data",
    "schema_mismatch",
    "database_unavailable",
    "database_timeout",
    "llm_timeout",
    "llm_rate_limited",
    "llm_parse_error",
    "llm_unavailable",
    "internal_error",
]


@dataclass(frozen=True)
class FailureInfo:
    """不直接暴露给用户的内部失败信息。"""

    category: FailureCategory
    code: str
    retryable: bool


def classify_exception(exc: BaseException, *, source: str = "") -> FailureInfo:
    """把底层异常映射为稳定的领域故障分类。"""
    del source
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "response", None)
    if hasattr(status_code, "status_code"):
        status_code = getattr(status_code, "status_code", None)

    if "database_timeout" in message:
        return FailureInfo("database_timeout", "database_timeout", True)
    if "database_unavailable" in message:
        return FailureInfo("database_unavailable", "database_unavailable", True)
    if "database_schema_mismatch" in message:
        return FailureInfo("schema_mismatch", "database_schema_mismatch", False)

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, SqlAlchemyTimeoutError)):
        if "sql" in name or "database" in message or isinstance(exc, SqlAlchemyTimeoutError):
            return FailureInfo("database_timeout", "database_timeout", True)
        return FailureInfo("llm_timeout", "llm_timeout", True)
    if isinstance(exc, (OperationalError, InterfaceError, DBAPIError)):
        if isinstance(exc, ProgrammingError) or any(
            marker in message for marker in ("undefined table", "undefined column", "does not exist", "schema")
        ):
            return FailureInfo("schema_mismatch", "database_schema_mismatch", False)
        return FailureInfo("database_unavailable", "database_unavailable", True)
    if isinstance(exc, ProgrammingError):
        return FailureInfo("schema_mismatch", "database_schema_mismatch", False)
    if status_code == 429 or "rate limit" in message or "ratelimit" in name:
        return FailureInfo("llm_rate_limited", "llm_rate_limited", True)
    if any(marker in name for marker in ("validationerror", "outputparser", "jsondecode", "jsondecode")):
        return FailureInfo("llm_parse_error", "llm_parse_error", True)
    if any(marker in name for marker in ("connection", "apierror", "serviceunavailable")):
        return FailureInfo("llm_unavailable", "llm_unavailable", True)
    if "sqlvalidationerror" in name or "unsafe" in message:
        return FailureInfo("sql_unsafe", "sql_unsafe", False)
    return FailureInfo("internal_error", "internal_error", False)


def classify_no_data() -> FailureInfo:
    """显式标记空结果，避免与数据库异常混淆。"""
    return FailureInfo("no_data", "no_data", False)


def classify_user_clarification() -> FailureInfo:
    """显式标记用户输入不足。"""
    return FailureInfo("user_clarification", "user_clarification", False)


def classify_unsupported(code: str = "unsupported") -> FailureInfo:
    """标记明确超出结构化财务库能力边界的问题。"""
    return FailureInfo("unsupported", code, False)


def classify_sql_validation(error_type: str) -> FailureInfo:
    """将规则门错误映射为安全或 Schema 故障。"""
    if error_type == "schema":
        return FailureInfo("schema_mismatch", "sql_schema_mismatch", False)
    return FailureInfo("sql_unsafe", f"sql_{error_type or 'validation'}", False)


__all__ = [
    "FailureCategory",
    "FailureInfo",
    "classify_exception",
    "classify_no_data",
    "classify_sql_validation",
    "classify_unsupported",
    "classify_user_clarification",
]
