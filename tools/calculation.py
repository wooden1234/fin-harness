"""受限金融计算工具。"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

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


class ResolvedCalculation(BaseModel):
    """已由受治理调用层从 Evidence 解析出的计算任务。"""

    model_config = ConfigDict(extra="forbid")

    calculation_id: str = Field(min_length=1, max_length=80)
    operation: CalculationOperation
    current_value: float
    reference_value: float
    periods: float = Field(default=1.0, gt=0)
    input_evidence_ids: list[str] = Field(min_length=1, max_length=2)
    operand_refs: list[dict[str, int | str]] = Field(min_length=2, max_length=2)


def _calculate(item: ResolvedCalculation) -> dict:
    """执行单项白名单计算；输入值只能来自上游 Evidence 解析结果。"""
    operation = item.operation
    current_value = item.current_value
    reference_value = item.reference_value
    periods = item.periods
    if operation in {
        "change_rate",
        "drawdown",
        "year_over_year",
        "quarter_over_quarter",
        "ratio",
    } and reference_value == 0:
        return {
            "ok": False,
            "calculation_id": item.calculation_id,
            "error": "division_by_zero",
        }

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
        if current_value < 0 or reference_value <= 0:
            return {
                "ok": False,
                "calculation_id": item.calculation_id,
                "error": "invalid_cagr_input",
            }
        value = ((current_value / reference_value) ** (1 / periods) - 1) * 100
        formula = "((current_value/reference_value)^(1/periods)-1)*100"
        unit = "%"

    return {
        "ok": True,
        "calculation_id": item.calculation_id,
        "operation": operation,
        "value": round(value, 6),
        "unit": unit,
        "formula": formula,
        "inputs": {
            "current_value": current_value,
            "reference_value": reference_value,
            "periods": periods,
        },
        "input_evidence_ids": list(dict.fromkeys(item.input_evidence_ids)),
        "operand_refs": item.operand_refs,
    }


@tool(parse_docstring=True)
async def run_calculation(
    calculations: list[ResolvedCalculation],
) -> dict:
    """批量执行已由服务端从 Evidence 解析的白名单金融计算。

    Args:
        calculations: 1-8 项已解析计算；MainAgent 不直接填写其中的数值。
    """
    if not calculations or len(calculations) > 8:
        return {"ok": False, "error": "invalid_calculation_batch_size"}
    results = [_calculate(item) for item in calculations]
    failures = [item for item in results if not item.get("ok")]
    if failures:
        return {
            "ok": False,
            "error": "calculation_batch_failed",
            "failed_calculations": failures,
        }
    return {"ok": True, "calculations": results}


register_tool(
    ToolSpec(
        tool_id="calculation.run",
        name="run_calculation",
        description=(
            "一次批量计算最多八项；MainAgent 只提交 Evidence ID 与 facts 下标，"
            "输入数值由服务端解析，禁止手算或按数值猜来源"
        ),
        read_only=True,
        timeout_seconds=2.0,
    ),
    langchain_tool=run_calculation,
)


__all__ = ["CalculationOperation", "ResolvedCalculation", "run_calculation"]
