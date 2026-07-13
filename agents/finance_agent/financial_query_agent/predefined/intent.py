"""predefined 白名单路径的标准查询意图。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class FinancialQueryIntent(BaseModel):
    """白名单模板执行使用的标准化查询意图。"""

    companies: list[str] = Field(
        default_factory=list,
        description="标准化后的公司候选",
    )
    years: list[int] = Field(
        default_factory=list,
        description="标准化后的年份或财年列表",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="标准化后的指标列表",
    )
    operation: Literal["lookup", "latest", "compare", "trend"] = Field(
        default="lookup",
        description="查询意图：精确查数、最新一期、公司对比、趋势查询",
    )
    time_scope: Literal["single", "latest", "range", "trailing_n_years", "unspecified"] = Field(
        default="unspecified",
        description="时间范围口径",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="最多返回多少条结果")
    ambiguity: list[dict[str, Any]] = Field(
        default_factory=list,
        description="标准化阶段识别出的歧义信息",
    )

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

    def has_template_blocking_ambiguity(self) -> bool:
        """判断当前意图是否仍然存在会阻断模板路由的歧义。"""
        return bool(self.ambiguity)


# 兼容旧调用方的历史名称。
FinancialFactQuery = FinancialQueryIntent


__all__ = ["FinancialFactQuery", "FinancialQueryIntent"]
