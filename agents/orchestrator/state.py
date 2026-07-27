"""Root Orchestrator V2 的 LangGraph 状态。"""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from app.shared import Citation, CoreState
from agents.orchestrator.contracts import (
    AgentResult,
    Evidence,
    QualityReport,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)


class OrchestratorState(CoreState):
    """顶层编排状态；专业子图通过标准契约与它通信。"""

    request_profile: NotRequired[RequestProfile]
    task_plan: NotRequired[TaskPlan]
    current_task: NotRequired[TaskSpec]
    current_dependency_results: NotRequired[list[AgentResult]]

    agent_results: NotRequired[Annotated[list[AgentResult], add]]
    evidence: NotRequired[Annotated[list[Evidence], add]]
    citations: NotRequired[Annotated[list[Citation], add]]
    quality_report: NotRequired[QualityReport]

    active_task_ids: NotRequired[list[str]]
    replan_count: NotRequired[int]
    next_action: NotRequired[str]
    execution_status: NotRequired[str]
    summary: NotRequired[str]
    route: NotRequired[str]
    guardrails_pass: NotRequired[bool]
    guardrails_reason: NotRequired[str]


__all__ = ["OrchestratorState"]
