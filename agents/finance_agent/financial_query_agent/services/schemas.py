"""financial_query_agent 领域模型与输出结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialFactQuery,
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.services.errors import FailureCategory


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
    canonical_code: str = Field(default="")
    raw_value: str = Field(default="")
    value: str = Field(default="")
    unit: str = Field(default="")
    currency: str = Field(default="")
    source: str = Field(default="")
    page_num: int | None = Field(default=None)
    doc_id: str = Field(default="")
    document_id: int | None = Field(default=None)
    table_id: int | None = Field(default=None)
    source_cell_id: int | None = Field(default=None)
    section: str = Field(default="")


class QueryContract(BaseModel):
    """SQL 结果应满足的查询语义契约。"""

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_contract(cls, value: Any) -> Any:
        """兼容模型返回的单数 canonical_code 和 ratio 操作。"""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if not payload.get("metrics"):
            legacy_metrics = payload.get("canonical_code") or payload.get("canonical_codes")
            if legacy_metrics:
                payload["metrics"] = legacy_metrics
        operation = str(payload.get("operation") or "unknown").strip().lower()
        if operation in {"ratio", "aggregate_ratio", "calculation"}:
            payload["operation"] = "aggregate"
        elif operation in {"point_query", "single", "single_query"}:
            payload["operation"] = "point_lookup"
        return payload

    companies: list[str] = Field(default_factory=list, description="期望的公司名、简称或股票代码")
    years: list[int] = Field(default_factory=list, description="期望的报告年份")
    metrics: list[str] = Field(default_factory=list, description="期望的 canonical_code 列表")
    period_type: Literal["annual", "quarter", "half_year", "unknown"] = Field(
        default="unknown",
        description="期望的期间类型",
    )
    operation: Literal["point_lookup", "list", "compare", "trend", "aggregate", "unknown"] = Field(
        default="unknown",
        description="查询操作类型",
    )

    @field_validator("metrics", mode="before")
    @classmethod
    def normalize_metrics(cls, value: Any) -> list[str]:
        """兼容模型返回的指标对象列表。"""
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("canonical_code") or item.get("code") or ""
            if item:
                normalized.append(str(item))
        return normalized

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, value: Any) -> str:
        """兼容旧 Prompt 生成的 query_single 枚举值。"""
        operation = str(value or "unknown").strip().lower()
        if operation in {"query_single", "lookup", "single_lookup"}:
            return "point_lookup"
        return operation


class GeneratedFinancialSql(BaseModel):
    """复杂查询生成的只读 SQL。"""

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_payload(cls, value: Any) -> Any:
        """兼容旧模型返回的 generated_sql 和 generate 路由。"""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if not payload.get("sql") and payload.get("generated_sql"):
            payload["sql"] = payload["generated_sql"]
        if payload.get("route") in {"generate", "default", "correct", "corrected", "fix"}:
            payload["route"] = "execute"
        return payload

    sql: str = Field(default="", description="只读 SELECT SQL，必须是单条语句。")
    params: dict[str, Any] = Field(default_factory=dict, description="SQL 命名参数。")
    reason: str = Field(default="", description="简述 SQL 的查询思路与口径。")
    route: Literal["execute", "clarify", "sql"] = Field(
        default="execute",
        description="若信息不足则要求补充，否则执行 SQL。",
    )
    missing_fields: list[str] = Field(default_factory=list, description="复杂查询仍缺失的字段。")
    query_contract: QueryContract | None = Field(
        default=None,
        description="SQL 执行结果必须满足的公司、年份、指标和期间契约。",
    )
    failure_category: FailureCategory | None = Field(
        default=None,
        description="仅供内部监控使用的生成失败分类，不直接展示给用户。",
    )
    failure_code: str = Field(default="", description="仅供内部监控使用的稳定错误码。")
    failure_retryable: bool = Field(default=False, description="该失败是否允许基础设施重试。")

    @field_validator("route", mode="before")
    @classmethod
    def normalize_route(cls, value: Any) -> str:
        route = str(value or "execute").strip().lower()
        if route in {"query", "select", "run", "execute_sql", "default", "generate"}:
            return "execute"
        return route


__all__ = [
    "FinancialFactQuery",
    "FinancialQueryIntent",
    "FinancialSqlResultRow",
    "GeneratedFinancialSql",
    "QueryContract",
]
