"""predefined 工具 schema，对应 assistgen kg_tools_list.predefined_cypher。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    VALID_TEMPLATE_IDS,
)


class predefined_sql(BaseModel):
    """白名单 SQL 模板工具：从用户问题中选择最合适的模板并提取执行参数。

    可选模板（template_id 必须精确匹配，槽位数量也必须精确匹配）：
    - exact_metric_lookup: 恰好 1 公司 + 恰好 1 年份 + 恰好 1 指标
    - latest_metric_lookup: 恰好 1 公司 + 恰好 1 指标，且未指定年份
    - compare_metric_lookup: ≥2 公司 + 恰好 1 显式共同年份 + 恰好 1 指标
    - compare_year_metric_lookup: 恰好 1 公司 + ≥2 显式年份 + 恰好 1 指标（单公司跨年对比）
    - trend_metric_lookup: 恰好 1 公司 + 恰好 1 指标 + ≥2 个显式年份（趋势叙述）

    """

    template_id: Literal[
        "exact_metric_lookup",
        "latest_metric_lookup",
        "compare_metric_lookup",
        "compare_year_metric_lookup",
        "trend_metric_lookup",
    ] = Field(..., description="白名单模板 ID")
    companies: list[str] = Field(
        default_factory=list,
        description="公司名、简称或股票代码列表；数量必须符合所选模板契约",
    )
    years: list[int] = Field(
        default_factory=list,
        description=(
            "报告年份或财年列表。"
            "exact/compare_metric_lookup 必须恰好 1 个；"
            "compare_year_metric_lookup/trend 必须 ≥2 个显式年份；"
            "latest 必须为 []；"
            "不要把「近三年」留空，无法展开则不要选跨年模板"
        ),
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="财务指标列表；所有模板都必须恰好 1 个已批准指标",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="最多返回多少条结果")

def is_valid_template_id(template_id: str) -> bool:
    return template_id in VALID_TEMPLATE_IDS


__all__ = ["is_valid_template_id", "predefined_sql"]
