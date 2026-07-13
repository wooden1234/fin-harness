"""四季度汇总后处理。"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    ResolvedMetricBinding,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)

QUARTER_LABELS = ("第一季度", "第二季度", "第三季度", "第四季度")


def _parse_numeric(value: str) -> Decimal | None:
    cleaned = (value or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def aggregate_quarter_rows(
    rows: list[FinancialSqlResultRow],
    bindings: list[ResolvedMetricBinding],
) -> list[FinancialSqlResultRow]:
    """对 sum_quarters 策略的 binding 将四季度明细汇总为年度值。"""
    if not rows or not bindings:
        return rows

    sum_bindings = {
        (binding.company_id, binding.metric_id): binding
        for binding in bindings
        if binding.selected_strategy == "sum_quarters"
    }
    if not sum_bindings:
        return rows

    annual_rows: list[FinancialSqlResultRow] = []
    quarter_groups: dict[tuple[int, int, int], list[FinancialSqlResultRow]] = {}

    for row in rows:
        company_id = row.company_id
        if company_id is None:
            annual_rows.append(row)
            continue
        key = (company_id, _metric_id_from_row(row, bindings), _target_year(row, bindings))
        binding = sum_bindings.get((company_id, key[1]))
        if binding is None or row.period_type != "quarter":
            annual_rows.append(row)
            continue
        quarter_groups.setdefault(key, []).append(row)

    aggregated: list[FinancialSqlResultRow] = []
    for key, group_rows in quarter_groups.items():
        company_id, metric_id, year = key
        labels = {row.period_label for row in group_rows}
        if not all(label in labels for label in QUARTER_LABELS):
            aggregated.extend(group_rows)
            continue
        total = Decimal(0)
        has_value = False
        for row in group_rows:
            numeric = _parse_numeric(row.value) or _parse_numeric(row.raw_value)
            if numeric is None:
                continue
            total += numeric
            has_value = True
        if not has_value:
            aggregated.extend(group_rows)
            continue
        sample = group_rows[0]
        aggregated.append(
            sample.model_copy(
                update={
                    "period_year": year,
                    "period_label": "四季度汇总",
                    "period_type": "annual",
                    "value": _format_decimal(total),
                    "raw_value": _format_decimal(total),
                }
            )
        )

    return annual_rows + aggregated


def _metric_id_from_row(row: FinancialSqlResultRow, bindings: list[ResolvedMetricBinding]) -> int:
    for binding in bindings:
        if binding.company_id == row.company_id and binding.metric_name == row.metric_name:
            return binding.metric_id
    for binding in bindings:
        if binding.company_id == row.company_id:
            return binding.metric_id
    return -1


def _target_year(row: FinancialSqlResultRow, bindings: list[ResolvedMetricBinding]) -> int:
    for binding in bindings:
        if binding.company_id == row.company_id and binding.selected_year is not None:
            return binding.selected_year
    return row.period_year or row.fiscal_year or 0


__all__ = ["aggregate_quarter_rows"]
