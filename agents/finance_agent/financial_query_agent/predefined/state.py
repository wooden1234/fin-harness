"""predefined 内部状态，对应 assistgen PredefinedCypherInputState / CypherOutputState。"""

from __future__ import annotations

from typing import Any, NotRequired
from typing_extensions import TypedDict

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)


class PredefinedSqlInputState(TypedDict):
    """tool_selection → execution 的输入，对应 assistgen PredefinedCypherInputState。"""

    task: str
    query_name: str
    query_parameters: dict[str, Any]
    intent: FinancialQueryIntent
    resolved_query: NotRequired[Any]
    steps: NotRequired[list[str]]


class PredefinedSqlOutputState(TypedDict):
    """execution 输出，与 text2cypher 结果结构对齐便于下游汇总。"""

    task: str
    template_id: str
    statement: str
    parameters: dict[str, Any]
    rows: list[FinancialSqlResultRow]
    missing_fields: list[str]
    errors: list[str]
    steps: list[str]


__all__ = [
    "PredefinedSqlInputState",
    "PredefinedSqlOutputState",
]
