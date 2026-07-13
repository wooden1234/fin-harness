"""financial_query_agent 领域模型与输出结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialFactQuery,
    FinancialQueryIntent,
)


class FinancialSqlResultRow(BaseModel):
    """统一的 SQL 查询结果行，用于格式化答案与引用。"""

    company_id: int | None = Field(default=None)
    company_name: str = Field(default="未知公司")
    ticker: str = Field(default="")
    fiscal_year: int | None = Field(default=None)
    period_year: int | None = Field(default=None)
    period_label: str = Field(default="")
    period_type: str = Field(default="")
    metric_name: str = Field(default="未知指标")
    raw_value: str = Field(default="")
    value: str = Field(default="")
    unit: str = Field(default="")
    currency: str = Field(default="")
    source: str = Field(default="")
    page_num: int | None = Field(default=None)
    doc_id: str = Field(default="")


class GeneratedFinancialSql(BaseModel):
    """复杂查询生成的只读 SQL。"""

    sql: str = Field(default="", description="只读 SELECT SQL，必须是单条语句。")
    params: dict[str, Any] = Field(default_factory=dict, description="SQL 命名参数。")
    reason: str = Field(default="", description="简述 SQL 的查询思路与口径。")
    route: Literal["execute", "clarify", "sql"] = Field(
        default="execute",
        description="若信息不足则要求补充，否则执行 SQL。",
    )
    missing_fields: list[str] = Field(default_factory=list, description="复杂查询仍缺失的字段。")

    @field_validator("route", mode="before")
    @classmethod
    def normalize_route(cls, value: Any) -> str:
        route = str(value or "execute").strip().lower()
        if route in {"query", "select", "run", "execute_sql"}:
            return "execute"
        return route


__all__ = [
    "FinancialFactQuery",
    "FinancialQueryIntent",
    "FinancialSqlResultRow",
    "GeneratedFinancialSql",
]
