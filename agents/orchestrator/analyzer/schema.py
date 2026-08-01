"""Analyzer 的语义输入与结构化输出契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.orchestrator.contracts import MarketQueryPlan

AnalyzerIntent = Literal[
    "concept_explain",
    "product_policy",
    "structured_metric",
    "document_qa",
    "market_query",
    "research_search",
    "stock_screening",
    "candidate_compute",
    "open_research",
    "entity_comparison",
    "general_chat",
    "clarify",
]
EntityScopeType = Literal["single", "explicit_group", "dynamic_group"]
PdfKnowledgeCategory = Literal[
    "annual_reports",
    "research_reports",
    "industry_whitepapers",
    "macro_research",
    "policy",
]


class DocumentLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = Field(default=None, max_length=80)
    title: str | None = Field(default=None, max_length=200)
    categories: list[PdfKnowledgeCategory] = Field(default_factory=list, max_length=5)
    source_locked: bool = False


class AnalyzerConstraints(BaseModel):
    """仅允许业务语义约束，禁止模型注入执行标识。"""

    model_config = ConfigDict(extra="forbid")

    time_range: str | None = Field(default=None, max_length=80)
    entity_scope_type: EntityScopeType | None = None
    analysis_dimensions: list[str] = Field(default_factory=list, max_length=12)
    comparison_basis: list[str] = Field(default_factory=list, max_length=12)
    candidate_set_id: str | None = Field(default=None, max_length=120)
    market_query_plan: MarketQueryPlan | None = None
    document: DocumentLocator | None = None
    semantic_history: bool = False

    @field_validator("entity_scope_type", mode="before")
    @classmethod
    def normalize_entity_scope_type(cls, value: Any) -> Any:
        """将模型常见的行业同义值收敛到稳定的范围协议。"""
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "industry": "dynamic_group",
            "sector": "dynamic_group",
            "theme": "dynamic_group",
        }
        return aliases.get(normalized, normalized)


class ActiveTopicProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str = ""
    domain: str = ""
    entities: list[str] = Field(default_factory=list, max_length=16)
    securities: list[str] = Field(default_factory=list, max_length=12)
    metrics: list[str] = Field(default_factory=list, max_length=16)
    time_ranges: list[str] = Field(default_factory=list, max_length=8)


class ArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    artifact_id: str = ""
    universe: str = ""
    as_of: str = ""
    row_count: int = Field(default=0, ge=0)
    available_fields: list[str] = Field(default_factory=list, max_length=64)
    source_task_id: str = ""


class AnalyzerInputEnvelope(BaseModel):
    """只投影决策所需上下文，不传摘要正文或候选数据行。"""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    rewritten_query: str = ""
    rewrite_status: str = "passthrough"
    resolved_entities: list[str] = Field(default_factory=list, max_length=16)
    active_topic: ActiveTopicProjection | None = None
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list, max_length=3)

    @property
    def effective_query(self) -> str:
        return self.rewritten_query.strip() or self.original_query.strip()


class AnalyzerOutput(BaseModel):
    """LLM 只输出语义，不得选择 Agent、Tool、来源或预算。"""

    model_config = ConfigDict(extra="forbid")

    normalized_query: str = ""
    intents: list[AnalyzerIntent] = Field(default_factory=list)
    freshness_required: bool = False
    entities: list[str] = Field(default_factory=list)
    constraints: AnalyzerConstraints = Field(default_factory=AnalyzerConstraints)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_message: str = Field(default="", max_length=1800)
    rationale: str = ""


__all__ = [
    "ActiveTopicProjection",
    "AnalyzerConstraints",
    "AnalyzerInputEnvelope",
    "AnalyzerIntent",
    "AnalyzerOutput",
    "ArtifactDescriptor",
    "DocumentLocator",
]
