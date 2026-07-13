"""text_to_sql 执行与结果整理节点。"""

from __future__ import annotations

import re
from typing import Any

from agents.finance_agent.financial_query_agent.predefined.semantic.company_resolver import (
    CompanyResolver,
)
from agents.finance_agent.financial_query_agent.services.fact_service import FinancialFactService
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow

_COMPANY_NAME_PARAM_RE = re.compile(r"\bcompany\.name\s+(?:=|in)\s*:(company_names?|companies)\b", re.IGNORECASE)
_COMPANY_KEY_PARAM_RE = re.compile(r"\bcompany\.company_key\s+(?:=|in)\s*:(company_keys?)\b", re.IGNORECASE)
_COMPANY_TICKER_PARAM_RE = re.compile(r"\bcompany\.ticker\s+(?:=|in)\s*:(tickers?)\b", re.IGNORECASE)


async def _resolve_company_values(values: list[str], *, target: str) -> list[str]:
    resolved = await CompanyResolver.resolve(values)
    if not resolved:
        return values
    by_input: dict[str, str] = {}
    for original in values:
        matches = await CompanyResolver.resolve([original])
        if not matches:
            by_input[original] = original
            continue
        company = matches[0]
        if target == "company_key":
            by_input[original] = company.db_company_key
        elif target == "ticker":
            by_input[original] = company.ticker or original
        else:
            by_input[original] = company.name
    return [by_input.get(value, value) for value in values]


async def _normalize_company_params(sql: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)

    async def normalize_key(param_name: str, target: str) -> None:
        if param_name not in normalized:
            return
        raw_value = normalized[param_name]
        if isinstance(raw_value, list):
            values = [str(item) for item in raw_value]
            normalized[param_name] = await _resolve_company_values(values, target=target)
            return
        if isinstance(raw_value, str):
            resolved_values = await _resolve_company_values([raw_value], target=target)
            normalized[param_name] = resolved_values[0] if resolved_values else raw_value

    for match in _COMPANY_NAME_PARAM_RE.finditer(sql):
        await normalize_key(match.group(1), "name")
    for match in _COMPANY_KEY_PARAM_RE.finditer(sql):
        await normalize_key(match.group(1), "company_key")
    for match in _COMPANY_TICKER_PARAM_RE.finditer(sql):
        await normalize_key(match.group(1), "ticker")
    return normalized


async def execute_generated_sql(
    sql: str,
    *,
    params: dict[str, Any] | None = None,
    limit: int = 5,
) -> list[FinancialSqlResultRow]:
    """执行已通过校验的 SQL，不在这里承担生成与修正逻辑。"""
    normalized_params = await _normalize_company_params(sql, params or {})
    return await FinancialFactService.run_generated_sql(
        sql,
        params=normalized_params,
        limit=limit,
    )


def _row_group_key(row: FinancialSqlResultRow) -> tuple[object, object, str] | None:
    company = row.company_id if row.company_id is not None else row.company_name.strip()
    year = row.period_year or row.fiscal_year
    metric = row.metric_name.strip()
    if not company or not year or not metric:
        return None
    return company, year, metric


def _is_full_year_label(label: str) -> bool:
    normalized = label.strip()
    if not normalized:
        return False
    date_markers = ("月", "日", "三十日", "三十一日", "quarter", "季度", "q1", "q2", "q3", "q4")
    return not any(marker in normalized.lower() for marker in date_markers)


def _row_quality_score(row: FinancialSqlResultRow) -> tuple[int, int, int, int]:
    year = row.period_year or row.fiscal_year
    same_fiscal_year = int(bool(row.fiscal_year and year and row.fiscal_year == year))
    annual_period = int(row.period_type in {"", "annual"})
    full_year_label = int(_is_full_year_label(row.period_label))
    has_value = int(bool(row.raw_value or row.value))
    page_penalty = -(row.page_num or 9999)
    return (
        same_fiscal_year * 100 + annual_period * 40 + full_year_label * 20 + has_value * 10,
        page_penalty,
        -(row.fiscal_year or 0),
        -(row.company_id or 0),
    )


def select_best_disclosure_rows(rows: list[FinancialSqlResultRow]) -> list[FinancialSqlResultRow]:
    """按公司、年份、指标保留最可信披露行，减少重复年报/对比期行。"""
    selected: dict[tuple[object, object, str], FinancialSqlResultRow] = {}
    passthrough: list[FinancialSqlResultRow] = []
    order: list[tuple[object, object, str]] = []
    for row in rows:
        key = _row_group_key(row)
        if key is None:
            passthrough.append(row)
            continue
        if key not in selected:
            selected[key] = row
            order.append(key)
            continue
        if _row_quality_score(row) > _row_quality_score(selected[key]):
            selected[key] = row
    return [selected[key] for key in order] + passthrough


def format_sql_rows(rows: list[FinancialSqlResultRow]) -> str:
    """统一封装 SQL 结果格式化，结构化查询只返回答案。"""
    deduped_rows = select_best_disclosure_rows(rows)
    if not deduped_rows:
        return "（数据库中未找到匹配的财务指标，建议改查 PDF 文档库。）"
    return FinancialFactService.format_sql_answer(deduped_rows)


__all__ = ["execute_generated_sql", "format_sql_rows", "select_best_disclosure_rows"]
