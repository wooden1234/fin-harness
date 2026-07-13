"""predefined 工具 schema，对应 assistgen kg_tools_list.predefined_cypher。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    VALID_TEMPLATE_IDS,
    template_catalog_text,
)


class predefined_sql(BaseModel):
    """白名单 SQL 模板工具：从用户问题中选择最合适的模板并提取执行参数。

    可选模板（template_id 必须精确匹配）：
    - exact_metric_lookup: 单公司 + 单年份 + 单指标精确查数
    - latest_metric_lookup: 单公司 + 单指标 + 最新一期查数
    - compare_metric_lookup: 多公司或多指标对比查询
    - trend_metric_lookup: 单公司单指标跨年份趋势查询

    详细说明：
    """

    template_id: Literal[
        "exact_metric_lookup",
        "latest_metric_lookup",
        "compare_metric_lookup",
        "trend_metric_lookup",
    ] = Field(..., description="白名单模板 ID")
    companies: list[str] = Field(
        default_factory=list,
        description="公司名、简称或股票代码列表",
    )
    years: list[int] = Field(
        default_factory=list,
        description="报告年份或财年列表；未提及则 []",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="财务指标列表，例如 ['营业收入']",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="最多返回多少条结果")


# 将 catalog 注入 docstring，供 LLM bind_tools 时阅读
predefined_sql.__doc__ = (predefined_sql.__doc__ or "") + "\n" + template_catalog_text()


def is_valid_template_id(template_id: str) -> bool:
    return template_id in VALID_TEMPLATE_IDS


__all__ = ["is_valid_template_id", "predefined_sql"]
