"""Main DeepAgent 的工具目录与参数契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MAIN_TOOL_IDS = (
    "weather.get",
    "web.search",
    "iwencai.query",
    "iwencai.screen",
    "iwencai.compare_entities",
    "iwencai.market.query",
    "iwencai.industry.query",
    "iwencai.index.query",
    "iwencai.rating.query",
    "iwencai.announcement.search",
    "iwencai.report.search",
    "iwencai.fund.screen",
    "knowledge.faq.search",
    "knowledge.pdf.catalog",
    "knowledge.pdf.search",
    "knowledge.fact.lookup",
    "calculation.run",
)


class MainFaqSearchArgs(BaseModel):
    """去除旧 ResearchPlan ID 后的 Main Agent FAQ 参数。"""

    query: str
    domain: Literal["capital_market", "corporate_finance"]
    top_k: int = Field(default=3, ge=1, le=5)


class MainPdfSearchArgs(BaseModel):
    """去除旧 ResearchPlan ID 后的 Main Agent PDF 参数。"""

    query: str
    categories: list[
        Literal[
            "annual_reports",
            "research_reports",
            "industry_whitepapers",
            "macro_research",
            "policy",
        ]
    ]
    doc_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=8)
    source_locked: bool = False


class MainWebSearchArgs(BaseModel):
    """Web 调用只传问句；域名范围由 tools.web_search 配置决定。"""

    query: str


class MainCompareEntitiesArgs(BaseModel):
    """多实体并发对比参数；实体数量在此处硬约束，避免模型传入单实体或过多实体。"""

    entities: list[str] = Field(min_length=2, max_length=6)
    query: str
    limit: int = Field(default=10, ge=1, le=20)


MAIN_TOOL_ARGS_SCHEMAS = {
    "web.search": MainWebSearchArgs,
    "knowledge.faq.search": MainFaqSearchArgs,
    "knowledge.pdf.search": MainPdfSearchArgs,
    "iwencai.compare_entities": MainCompareEntitiesArgs,
}


__all__ = ["MAIN_TOOL_ARGS_SCHEMAS", "MAIN_TOOL_IDS"]
