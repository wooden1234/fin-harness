"""Main DeepAgent 的工具目录与参数契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from tools.calculation import CalculationOperation


MAIN_TOOL_IDS = (
    "weather.get",
    "web.search",
    "iwencai.query",
    "iwencai.finance.query",
    "iwencai.screen",
    "iwencai.usstock.screen",
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
    """Web 调用必须声明目标主体；域名范围由工具配置决定。"""

    query: str = Field(
        min_length=2,
        max_length=120,
        description=(
            "已润色的检索问句：主体+事件+时间/年份；"
            "一槽位一句，勿塞买卖动作话术，勿把多主题糊成超长句。"
        ),
    )
    entities: list[str] = Field(
        min_length=1,
        max_length=6,
        description="查询目标主体（公司/主题），用于结果相关性过滤。",
    )


class MainEvidenceFactRef(BaseModel):
    """指向 Journal 中某条 Evidence 的规范化事实。"""

    evidence_id: str = Field(min_length=1, max_length=120)
    fact_index: int = Field(ge=0)


class MainCalculationItem(BaseModel):
    """模型只声明运算和事实引用，不直接提供数值。"""

    calculation_id: str = Field(min_length=1, max_length=80)
    operation: CalculationOperation
    current: MainEvidenceFactRef
    reference: MainEvidenceFactRef
    periods: float = Field(default=1.0, gt=0)


class MainCalculationBatchArgs(BaseModel):
    """单次最多提交八项计算，避免并发调用消耗工具轮次。"""

    calculations: list[MainCalculationItem] = Field(min_length=1, max_length=8)


MAIN_TOOL_ARGS_SCHEMAS = {
    "web.search": MainWebSearchArgs,
    "knowledge.faq.search": MainFaqSearchArgs,
    "knowledge.pdf.search": MainPdfSearchArgs,
    "calculation.run": MainCalculationBatchArgs,
}


__all__ = ["MAIN_TOOL_ARGS_SCHEMAS", "MAIN_TOOL_IDS"]
