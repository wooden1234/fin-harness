"""独立金融研究工作流的内部状态。"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from agents.orchestrator.contracts import AgentResult, TaskSpec
from agents.research_workflow.contracts import (
    QuestionEvidenceAssessment,
    ResearchContextSummary,
    ResearchPlan,
)


class ResearchWorkflowState(TypedDict, total=False):
    """状态仅在研究子图内部流转，不暴露给 Root 编排器。"""

    messages: list[Any]
    query: str
    task_input: dict[str, Any]
    task_identity: dict[str, Any]
    dependency_results: list[AgentResult]
    research_plan: ResearchPlan
    source_tasks: list[TaskSpec]
    source_results: list[AgentResult]
    research_context_summary: ResearchContextSummary
    model_dependency_results: list[AgentResult]
    research_context_counters: dict[str, int]
    research_context_admitted: bool
    deep_result: AgentResult
    question_evidence_assessments: list[QuestionEvidenceAssessment]
    result: AgentResult


__all__ = ["ResearchWorkflowState"]
