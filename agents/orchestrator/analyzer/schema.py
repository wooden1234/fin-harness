"""Analyzer LLM 的结构化输出契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AnalyzerIntent = Literal[
    "stock_screening",
    "financial_analysis",
    "financial_research",
    "general_chat",
    "clarify",
]
AnalyzerComplexity = Literal["simple", "single_capability", "compound"]
PreferredAgent = Literal[
    "stock_screening_agent",
    "finance_agent",
    "general_agent",
]


class AnalyzerOutput(BaseModel):
    """LLM 请求画像；不含 TaskPlan，仅描述语义理解结果。"""

    normalized_query: str = ""
    intents: list[AnalyzerIntent] = Field(default_factory=list)
    complexity: AnalyzerComplexity = "simple"
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
