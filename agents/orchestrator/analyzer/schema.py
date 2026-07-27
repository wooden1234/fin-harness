"""Analyzer LLM 的结构化输出契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.orchestrator.contracts import DataSourceType, OperationType

AnalyzerIntent = Literal[
    "stock_screening",
    "market_query",
    "market_compute",
    "research_search",
    "financial_analysis",
    "financial_research",
    "deep_research",
    "general_chat",
    "clarify",
]
AnalyzerComplexity = Literal["simple", "single_capability", "compound"]
PreferredAgent = Literal[
    "market_acquisition_workflow",
    "research_retrieval_workflow",
    "research_workflow",
    "stock_screening_agent",
    "finance_agent",
    "general_agent",
    "market.compute",
]


class AnalyzerOutput(BaseModel):
    """LLM 请求画像；不含 TaskPlan，仅描述语义理解结果。"""

    normalized_query: str = ""
    intents: list[AnalyzerIntent] = Field(default_factory=list)
    complexity: AnalyzerComplexity = "simple"
    data_sources: list[DataSourceType] = Field(default_factory=list)
    operation_type: OperationType = "answer"
    preferred_agent: PreferredAgent | None = None
    freshness_required: bool = False
    entities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    rationale: str = ""


__all__ = [
    "AnalyzerComplexity",
    "AnalyzerIntent",
    "AnalyzerOutput",
    "PreferredAgent",
]
