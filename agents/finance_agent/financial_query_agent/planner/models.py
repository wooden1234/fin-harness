"""financial_query_agent 内部规划模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FinancialQueryPlan(BaseModel):
    """表示结构化查询内部的下一步规划结果。"""

    route: Literal["predefined", "text_to_sql"] = Field(
        default="text_to_sql",
        description="下一步进入白名单模板工作流或复杂 SQL 工作流。",
    )
    reason: str = Field(default="", description="简短说明为何做出当前规划。")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="规划置信度。")


__all__ = ["FinancialQueryPlan"]
