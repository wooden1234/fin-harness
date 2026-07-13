"""text_to_sql 子图内部状态。"""

from __future__ import annotations

from typing import Any, NotRequired
from typing_extensions import Literal, TypedDict

from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow

TextToSqlNextStep = Literal[
    "clarify_before_generate",
    "generate_sql",
    "clarify_after_generate",
    "validate_sql",
    "correct_sql",
    "clarify_after_correct",
    "execute_sql",
    "clarify_output",
    "unsafe_output",
    "execution_error_output",
    "format_output",
    "end",
]

TextToSqlFinalStatus = Literal[
    "clarify",
    "unsafe",
    "execution_error",
    "success",
]


class TextToSqlState(TypedDict):
    """text_to_sql 子图在重试循环中使用的局部状态。"""

    question: str
    schema_prompt: NotRequired[str]
    fewshot_examples: NotRequired[str]
    top_k: NotRequired[int]
    max_attempts: NotRequired[int]
    sql: NotRequired[str]
    sql_params: NotRequired[dict[str, Any]]
    sql_route: NotRequired[str]
    route_reason: NotRequired[str]
    missing_fields: NotRequired[list[str]]
    validated_sql: NotRequired[str]
    validation_error: NotRequired[str]
    validation_error_type: NotRequired[str]
    validation_errors: NotRequired[list[str]]
    validation_error_types: NotRequired[list[str]]
    next_step: NotRequired[TextToSqlNextStep]
    attempts: NotRequired[int]
    rows: NotRequired[list[FinancialSqlResultRow]]
    execution_error: NotRequired[str]
    halted: NotRequired[bool]
    halt_reason: NotRequired[str]
    halt_answer: NotRequired[str]
    answer: NotRequired[str]
    final_status: NotRequired[TextToSqlFinalStatus]


__all__ = ["TextToSqlFinalStatus", "TextToSqlNextStep", "TextToSqlState"]
