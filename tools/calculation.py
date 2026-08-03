"""受限金融计算工具。"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from tools.core.base import ToolSpec
from tools.core.registry import register_tool

CalculationOperation = Literal[
    "change_rate",
    "drawdown",
    "year_over_year",
    "quarter_over_quarter",
    "ratio",
    "difference",
    "cagr",
]


@tool(parse_docstring=True)
async def run_calculation(
    operation: CalculationOperation,
    current_value: float,
    reference_value: float,
    periods: float = 1.0,
    input_evidence_ids: list[str] | None = None,
) -> dict:
    """执行白名单金融计算，禁止传入任意表达式。

    Args:
        operation: 计算类型。
        current_value: 当前值或期末值。
        reference_value: 基准值、期初值或分母。
        periods: CAGR 的期数，其他计算传 1。
        input_evidence_ids: 计算输入对应的 Evidence ID。
    """
    evidence_ids = list(dict.fromkeys(input_evidence_ids or []))
    if not evidence_ids:
        return {"ok": False, "error": "input_evidence_required"}
    if operation in {
        "change_rate",
        "drawdown",
        "year_over_year",
        "quarter_over_quarter",
        "ratio",
    } and reference_value == 0:
        return {"ok": False, "error": "division_by_zero"}

    formula = ""
    if operation in {"change_rate", "year_over_year", "quarter_over_quarter"}:
        value = (current_value - reference_value) / abs(reference_value) * 100
        formula = "(current_value-reference_value)/abs(reference_value)*100"
        unit = "%"
    elif operation == "drawdown":
        value = (current_value - reference_value) / reference_value * 100
        formula = "(current_value-reference_value)/reference_value*100"
        unit = "%"
    elif operation == "ratio":
        value = current_value / reference_value
        formula = "current_value/reference_value"
        unit = ""
    elif operation == "difference":
        value = current_value - reference_value
        formula = "current_value-reference_value"
        unit = ""
    else:
        if current_value < 0 or reference_value <= 0 or periods <= 0:
            return {"ok": False, "error": "invalid_cagr_input"}
        value = ((current_value / reference_value) ** (1 / periods) - 1) * 100
        formula = "((current_value/reference_value)^(1/periods)-1)*100"
        unit = "%"

    return {
        "ok": True,
        "operation": operation,
        "value": round(value, 6),
        "unit": unit,
        "formula": formula,
        "inputs": {
            "current_value": current_value,
            "reference_value": reference_value,
            "periods": periods,
        },
        "input_evidence_ids": evidence_ids,
    }


register_tool(
    ToolSpec(
        tool_id="calculation.run",
        name="run_calculation",
        description="使用白名单公式计算涨跌幅、回撤、同比、环比、比率、差值或 CAGR",
        read_only=True,
        timeout_seconds=2.0,
    ),
    langchain_tool=run_calculation,
)


__all__ = ["run_calculation"]
