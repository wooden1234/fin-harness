"""独立金融研究工作流的内部状态。"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from agents.orchestrator.contracts import AgentResult, TaskSpec
from agents.research_workflow.contracts import ResearchPlan


class ResearchWorkflowState(TypedDict, total=False):
    """状态仅在研究子图内部流转，不暴露给 Root 编排器。"""

    messages: list[Any]
    query: str
    task_input: dict[str, Any]
    dependency_results: list[AgentResult]
    research_plan: ResearchPlan
    source_tasks: list[TaskSpec]
    source_results: list[AgentResult]
    deep_result: AgentResult
    result: AgentResult


__all__ = ["ResearchWorkflowState"]
