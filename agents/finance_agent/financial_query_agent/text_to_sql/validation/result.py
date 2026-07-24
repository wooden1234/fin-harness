"""text_to_sql 查询结果校验。"""

from __future__ import annotations

from dataclasses import dataclass
import re

from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.services.schemas import QueryContract
from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import CompanyResolver
from agents.finance_agent.financial_query_agent.text_to_sql.validation.node import ValidationErrorType
from agents.finance_agent.financial_query_agent.services.errors import FailureCategory

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
_CANONICAL_METRIC_ALIASES = {
    "营业收入": "REVENUE",
    "营收": "REVENUE",
    "收入": "REVENUE",
    "归属于上市公司股东的净利润": "NET_INCOME_ATTR_PARENT",
    "归母净利润": "NET_INCOME_ATTR_PARENT",
    "净利润": "NET_INCOME_ATTR_PARENT",
    "营业利润": "OPERATING_PROFIT",
    "经营利润": "OPERATING_PROFIT",
    "经营活动产生的现金流量净额": "OPERATING_CASHFLOW_NET",
    "经营现金流净额": "OPERATING_CASHFLOW_NET",
    "研发费用": "RND_EXPENSE",
    "研发支出": "RND_EXPENSE",
    "毛利率": "GROSS_MARGIN",
    "总资产": "TOTAL_ASSETS",
    "资产总计": "TOTAL_ASSETS",
    "总负债": "TOTAL_LIABILITIES",
    "负债合计": "TOTAL_LIABILITIES",
    "基本每股收益": "EPS_BASIC",
    "eps": "EPS_BASIC",
}


@dataclass(frozen=True)
class ResultValidation:
    ok: bool
    error: str = ""
    error_type: ValidationErrorType = ""
    should_clarify: bool = False
    failure_category: FailureCategory | None = None
    failure_code: str = ""
    failure_retryable: bool = False

    @classmethod
    def passed(
        cls,
        *,
        failure_category: FailureCategory | None = None,
        failure_code: str = "",
        failure_retryable: bool = False,
    ) -> ResultValidation:
        return cls(
            ok=True,
            failure_category=failure_category,
            failure_code=failure_code,
            failure_retryable=failure_retryable,
        )


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


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.strip().lower())


def _canonical_company(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    return _normalize_text(
        CompanyResolver._canonical_from_db_row(
            name=value,
            company_key=value,
            ticker=value,
        )
    )


def _canonical_metric(value: str) -> str:
    normalized = _normalize_text(value)
    return _CANONICAL_METRIC_ALIASES.get(normalized, normalized)


def _validate_query_contract(
    contract: QueryContract,
    rows: list[FinancialSqlResultRow],
) -> ResultValidation:
    if contract.companies:
        expected_companies = {_canonical_company(item) for item in contract.companies}
        expected_companies.discard("")
        actual_companies = {
            _canonical_company(
                row.company_name
                if row.company_name.strip() and row.company_name != _PLACEHOLDER_COMPANY
                else row.ticker
            )
            for row in rows
            if row.company_name.strip() or row.ticker.strip()
        }
        actual_companies.discard("")
        if expected_companies and not actual_companies.issubset(expected_companies):
            return ResultValidation(
                ok=False,
                error="结果校验失败：返回结果包含用户未查询的公司。",
                error_type="semantic",
            )
        if contract.operation in {"point_lookup", "compare", "trend"} and not expected_companies.issubset(actual_companies):
            return ResultValidation(
                ok=False,
                error="结果校验失败：返回结果缺少用户查询的公司。",
                error_type="semantic",
            )

    if contract.years:
        expected_years = set(contract.years)
        actual_years = {
            row.period_year or row.fiscal_year
            for row in rows
            if row.period_year or row.fiscal_year
        }
        if actual_years != expected_years:
            return ResultValidation(
                ok=False,
                error="结果校验失败：返回结果的年份与用户查询年份不一致。",
                error_type="semantic",
            )

    if contract.period_type != "unknown":
        actual_periods = {row.period_type for row in rows if row.period_type}
        if actual_periods and actual_periods != {contract.period_type}:
            return ResultValidation(
                ok=False,
                error="结果校验失败：返回结果的期间类型与用户查询口径不一致。",
                error_type="semantic",
            )

    if contract.metrics:
        expected_metrics = {_canonical_metric(item) for item in contract.metrics}
        actual_metrics = {
            _canonical_metric(row.canonical_code or row.metric_name)
            for row in rows
            if row.canonical_code or row.metric_name
        }
        if actual_metrics and not actual_metrics.issubset(expected_metrics):
            return ResultValidation(
                ok=False,
                error="结果校验失败：返回结果的财务指标与用户查询指标不一致。",
                error_type="semantic",
            )

    return ResultValidation.passed()


def validate_query_result(
    *,
    question: str,
    sql: str,
    rows: list[FinancialSqlResultRow],
    contract: QueryContract | None = None,
) -> ResultValidation:
    """结果层弱校验：空结果可疑、缺列/缺值、财务事实查数结果不完整。"""
    normalized_question = question.strip()
    normalized_sql = sql.strip()
    fact_value_query = _is_fact_value_query(normalized_sql)

    if not rows:
        if contract and contract.operation == "point_lookup":
            return ResultValidation(
                ok=False,
                error="结果校验失败：点查类查询没有返回结果。",
                error_type="result_empty",
            )
        if fact_value_query and _is_point_lookup_question(normalized_question):
            return ResultValidation(
                ok=False,
                error="结果校验失败：点查类财务问题返回 0 行，可能是公司/年份/指标口径或 JOIN 条件不正确。",
                error_type="result_empty",
            )
        return ResultValidation.passed()

    if contract:
        contract_validation = _validate_query_contract(contract, rows)
        if not contract_validation.ok:
            return contract_validation

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
    contract: QueryContract | None = None,
    config: object = None,
) -> ResultValidation:
    """先走规则校验，通过后再按开关做 LLM 结果质检。"""
    validation = validate_query_result(
        question=question,
        sql=sql,
        rows=rows,
        contract=contract,
    )
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
