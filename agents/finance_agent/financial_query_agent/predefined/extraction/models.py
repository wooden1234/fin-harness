"""predefined 白名单槽位抽取模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class PredefinedSlotExtraction(BaseModel):
    """白名单模板执行所需的最小槽位集合。"""

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
    operation: Literal["lookup", "latest", "compare", "compare_year", "trend"] = Field(
        default="lookup",
        description="查询意图：精确查数、最新一期、多公司对比、单公司跨年对比、趋势查询",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="最多返回多少条结果")

    @computed_field
    @property
    def company(self) -> str:
        return self.companies[0] if self.companies else ""

    @computed_field
    @property
    def year(self) -> int | None:
        return self.years[0] if self.years else None

    @computed_field
    @property
    def metric(self) -> str:
        return self.metrics[0] if self.metrics else ""


__all__ = ["PredefinedSlotExtraction"]
