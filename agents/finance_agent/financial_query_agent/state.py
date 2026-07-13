"""Financial Query Agent 领域 State：结构化财务查询子图内部字段。

属于 financial_query_agent 子图的字段集中在此，
父级 finance_agent 的通用字段在 finance_agent/state.py 里。
"""

from __future__ import annotations

from typing import Any, NotRequired
from typing_extensions import TypedDict

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)


class FinancialQueryState(TypedDict):
    """financial_query 子图内部读写状态"""
    financial_query_text: NotRequired[str]
    financial_query_intent: NotRequired[FinancialQueryIntent]
    financial_query_plan_route: NotRequired[str]
    financial_query_plan_reason: NotRequired[str]
    financial_query_missing_fields: NotRequired[list[str]]
    financial_query_sql: NotRequired[str]
    financial_query_validated_sql: NotRequired[str]
    financial_query_validation_error: NotRequired[str]
    financial_query_validation_errors: NotRequired[list[str]]
    financial_query_sql_attempts: NotRequired[int]
    financial_query_next_action_sql: NotRequired[str]
    financial_query_sql_params: NotRequired[dict[str, Any]]
    financial_query_template_id: NotRequired[str | None]
    financial_query_schema_prompt: NotRequired[str]
    financial_query_fewshot_examples: NotRequired[str]
