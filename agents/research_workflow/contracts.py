"""Research Workflow 内部使用的稳定研究计划契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.orchestrator.contracts import TaskSpec


class ResearchQuestion(BaseModel):
    """研究计划中的一个待回答问题。"""

    question_id: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=500)
    evidence_requirements: list[str] = Field(default_factory=list, max_length=8)


class ResearchPlanDraft(BaseModel):
    """LLM 只能提出研究问题和数据范围，不得指定 Agent 或 Tool。"""

    data_sources: list[str] = Field(default_factory=list, max_length=5)
    questions: list[ResearchQuestion] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", max_length=500)


class ResearchPlan(BaseModel):
    """围绕一个问题生成的内部研究计划。"""

    schema_version: str = "1.0"
    query: str
    entities: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    source_tasks: list[TaskSpec] = Field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ResearchPlan", "ResearchPlanDraft", "ResearchQuestion"]
