"""text_to_sql 校验节点。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from typing_extensions import Literal

from agents.finance_agent.financial_query_agent.services.sql_executor import FinancialSqlExecutor, SqlValidationError

ValidationErrorType = Literal["", "safety", "schema", "parameter", "semantic"]

_FACT_TABLE_RE = re.compile(r"\bfin_core\.annual_financial_facts\b", re.IGNORECASE)
_FACT_VALUE_RE = re.compile(
    r"\b(?:fact\.)?(?:value|raw_value)\b|\border\s+by\s+[^;]*\bfact\.value\b",
    re.IGNORECASE,
)
_SOURCE_METRIC_FILTER_RE = re.compile(
    r"\bmetric\.canonical_name\s*(?:=|in|like|ilike)\b",
    re.IGNORECASE,
)
_CANONICAL_PATH_RE = re.compile(
    r"\bfact\.canonical_code\b|\bcompany_metric_mappings\b|\bmapping\.canonical_code\b",
    re.IGNORECASE,
)
_MAPPING_TABLE_RE = re.compile(r"\bcompany_metric_mappings\b", re.IGNORECASE)
_MAPPING_APPROVED_RE = re.compile(r"\breview_status\s*=\s*'approved'", re.IGNORECASE)
_MAPPING_ACTIVE_RE = re.compile(r"\bis_active\s*=\s*true\b", re.IGNORECASE)


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
    if error.startswith("语义校验失败"):
        return "semantic"
    return "safety"


def _is_fact_value_query(sql: str) -> bool:
    if not _FACT_TABLE_RE.search(sql):
        return False
    return bool(_FACT_VALUE_RE.search(sql))


def _validate_financial_semantics(sql: str) -> None:
    if not _is_fact_value_query(sql):
        return
    if not _CANONICAL_PATH_RE.search(sql):
        raise SqlValidationError(
            "语义校验失败：财务事实查数必须使用 fact.canonical_code 或 JOIN fin_core.company_metric_mappings，"
            "不能只依赖 financial_metrics.canonical_name 作为统一指标口径。"
        )
    if _SOURCE_METRIC_FILTER_RE.search(sql) and "canonical_code" not in sql.lower():
        raise SqlValidationError(
            "语义校验失败：metric.canonical_name 只能作为源指标展示或辅助字段，不能单独作为财务指标口径过滤。"
        )
    if _MAPPING_TABLE_RE.search(sql):
        missing_rules: list[str] = []
        if not _MAPPING_ACTIVE_RE.search(sql):
            missing_rules.append("mapping.is_active = true")
        if not _MAPPING_APPROVED_RE.search(sql):
            missing_rules.append("mapping.review_status = 'approved'")
        if missing_rules:
            raise SqlValidationError(
                "语义校验失败：使用 company_metric_mappings 时必须包含 "
                + " 和 ".join(missing_rules)
                + "。"
            )


def validate_generated_sql(
    sql: str,
    *,
    params: dict[str, Any] | None = None,
) -> SqlValidationResult:
    """当前先复用白名单校验，后续可在此叠加语义与 Schema 校验。"""
    try:
        validated_sql = FinancialSqlExecutor.validate_readonly_sql(sql, params=params)
        _validate_financial_semantics(validated_sql)
        return SqlValidationResult(validated_sql=validated_sql)
    except SqlValidationError as exc:
        error = str(exc)
        return SqlValidationResult(
            validated_sql="",
            error=error,
            error_type=_classify_validation_error(error),
        )


__all__ = ["SqlValidationResult", "ValidationErrorType", "validate_generated_sql"]
