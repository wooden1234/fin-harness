"""text_to_sql 查询结果校验。"""

from __future__ import annotations

from dataclasses import dataclass
import re

from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.text_to_sql.validation.node import ValidationErrorType

_FACT_TABLE_RE = re.compile(r"\bfin_core\.annual_financial_facts\b", re.IGNORECASE)
_FACT_VALUE_RE = re.compile(
    r"\b(?:fact\.)?(?:value|raw_value)\b|\bAS\s+(?:value|raw_value)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_LIST_QUESTION_RE = re.compile(
    r"(哪些|列出|名单|排名|前十|多少家|有哪些|列举|清单|排行)",
)
_METRIC_KEYWORD_RE = re.compile(
    r"(营收|收入|利润|净利润|归母|资产|负债|现金流|毛利率|每股收益|研发费用|毛利|净资产|负债率)",
)
_COMPANY_HINT_RE = re.compile(
    r"([\u4e00-\u9fff]{2,}|[A-Za-z]{2,})",
)
_PLACEHOLDER_METRIC = "未知指标"
_PLACEHOLDER_COMPANY = "未知公司"


@dataclass(frozen=True)
class ResultValidation:
    ok: bool
    error: str = ""
    error_type: ValidationErrorType = ""
    should_clarify: bool = False

    @classmethod
    def passed(cls) -> ResultValidation:
        return cls(ok=True)


def _is_fact_value_query(sql: str) -> bool:
    if not _FACT_TABLE_RE.search(sql):
        return False
    return bool(_FACT_VALUE_RE.search(sql))


def _is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION_RE.search(question))


def _is_point_lookup_question(question: str) -> bool:
    if _is_list_question(question):
        return False
    has_metric = bool(_METRIC_KEYWORD_RE.search(question))
    has_year = bool(_YEAR_RE.search(question))
    has_entity = bool(_COMPANY_HINT_RE.search(question))
    return has_metric and (has_year or has_entity)


def _rows_have_metric_values(rows: list[FinancialSqlResultRow]) -> bool:
    for row in rows:
        if str(row.raw_value or "").strip() or str(row.value or "").strip():
            return True
    return False


def _rows_have_metric_names(rows: list[FinancialSqlResultRow]) -> bool:
    for row in rows:
        metric_name = str(row.metric_name or "").strip()
        if metric_name and metric_name != _PLACEHOLDER_METRIC:
            return True
    return False


def validate_query_result(
    *,
    question: str,
    sql: str,
    rows: list[FinancialSqlResultRow],
) -> ResultValidation:
    """结果层弱校验：空结果可疑、缺列/缺值、财务事实查数结果不完整。"""
    normalized_question = question.strip()
    normalized_sql = sql.strip()
    fact_value_query = _is_fact_value_query(normalized_sql)

    if not rows:
        if fact_value_query and _is_point_lookup_question(normalized_question):
            return ResultValidation(
                ok=False,
                error="结果校验失败：点查类财务问题返回 0 行，可能是公司/年份/指标口径或 JOIN 条件不正确。",
                error_type="result_empty",
            )
        return ResultValidation.passed()

    if fact_value_query:
        if not _rows_have_metric_values(rows):
            return ResultValidation(
                ok=False,
                error="结果校验失败：财务事实查询缺少有效数值列（value/raw_value 为空）。",
                error_type="result_schema",
            )
        if _is_point_lookup_question(normalized_question) and not _rows_have_metric_names(rows):
            return ResultValidation(
                ok=False,
                error="结果校验失败：财务事实查询缺少可识别的指标名称。",
                error_type="result_schema",
            )

    return ResultValidation.passed()


async def validate_query_result_full(
    *,
    question: str,
    sql: str,
    rows: list[FinancialSqlResultRow],
    config: object = None,
) -> ResultValidation:
    """先走规则校验，通过后再按开关做 LLM 结果质检。"""
    validation = validate_query_result(question=question, sql=sql, rows=rows)
    if not validation.ok:
        return validation

    from agents.finance_agent.financial_query_agent.text_to_sql.validation.llm_result import (
        validate_query_result_with_llm,
    )

    return await validate_query_result_with_llm(
        question=question,
        sql=sql,
        rows=rows,
        config=config,
    )


__all__ = ["ResultValidation", "validate_query_result", "validate_query_result_full"]
