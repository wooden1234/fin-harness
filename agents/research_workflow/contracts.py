"""Research Workflow 内部使用的稳定研究计划契约。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.orchestrator.contracts import EvidencePolicy, TaskSpec


class ResearchQuestionDraft(BaseModel):
    """LLM 可提出的研究问题，不包含 Agent、任务 ID 或执行策略。"""

    question_id: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=500)
    data_sources: list[str] = Field(default_factory=list, max_length=5)
    evidence_requirements: list[str] = Field(default_factory=list, max_length=8)


class ResearchQuestion(ResearchQuestionDraft):
    """已映射到可执行任务和证据标准的研究问题。"""

    source_task_ids: list[str] = Field(default_factory=list, max_length=8)
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)


class QuestionEvidenceAssessment(BaseModel):
    """一个研究问题对其来源任务证据的确定性验收结果。"""

    question_id: str = Field(min_length=1, max_length=64)
    source_task_ids: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(default_factory=list)
    passed: bool = False
    evidence_count: int = Field(default=0, ge=0)
    required_count: int = Field(default=0, ge=0)
    missing_provenance_count: int = Field(default=0, ge=0)
    structured_data_present: bool = False
    gaps: list[str] = Field(default_factory=list)


class ResearchPlanDraft(BaseModel):
    """LLM 只能提出研究问题和数据范围，不得指定 Agent 或 Tool。"""

    data_sources: list[str] = Field(default_factory=list, max_length=5)
    questions: list[ResearchQuestionDraft] = Field(
        default_factory=list,
        max_length=8,
    )
    rationale: str = Field(default="", max_length=500)


class ResearchPlan(BaseModel):
    """围绕一个问题生成的内部研究计划。"""

    schema_version: str = "2.0"
    query: str
    entities: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    source_tasks: list[TaskSpec] = Field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchFinding(BaseModel):
    """研究过程中的结论及其证据引用。"""

    claim: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=16)


class ResearchContextSummary(BaseModel):
    """只在单次 Research Workflow 内使用的结构化工作摘要。"""

    objective: str = Field(min_length=1, max_length=500)
    completed_questions: list[str] = Field(default_factory=list, max_length=16)
    pending_questions: list[str] = Field(default_factory=list, max_length=16)
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=24)
    conflicts: list[str] = Field(default_factory=list, max_length=12)
    failed_sources: list[str] = Field(default_factory=list, max_length=12)
    next_actions: list[str] = Field(default_factory=list, max_length=12)
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)


__all__ = [
    "QuestionEvidenceAssessment",
    "ResearchPlan",
    "ResearchPlanDraft",
    "ResearchQuestion",
    "ResearchQuestionDraft",
    "ResearchContextSummary",
    "ResearchFinding",
]
