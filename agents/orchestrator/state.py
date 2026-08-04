"""Root Orchestrator V2 的 LangGraph 状态。"""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from app.shared import Citation, CoreState
from agents.orchestrator.contracts import (
    AgentResult,
    Claim,
    ClaimEvidenceLink,
    ConstrainedAnswer,
    Evidence,
    EvidenceAssessment,
    EvidenceConflict,
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
    prior_agent_results: NotRequired[list[AgentResult]]
    memory_requirements: NotRequired[dict[str, dict[str, object]]]
    task_memory_context: NotRequired[dict[str, dict[str, object]]]
    current_task_memory_context: NotRequired[dict[str, object]]

    agent_results: NotRequired[Annotated[list[AgentResult], add]]
    evidence: NotRequired[Annotated[list[Evidence], add]]
    claims: NotRequired[list[Claim]]
    claim_evidence_links: NotRequired[list[ClaimEvidenceLink]]
    evidence_assessments: NotRequired[list[EvidenceAssessment]]
    evidence_conflicts: NotRequired[list[EvidenceConflict]]
    constrained_answer: NotRequired[ConstrainedAnswer]
    citations: NotRequired[Annotated[list[Citation], add]]
    quality_report: NotRequired[QualityReport]
    main_agent_response: NotRequired[object]
    main_agent_journal: NotRequired[dict[str, object]]
    main_quality_metrics: NotRequired[dict[str, object]]
    investment_action_sensitive: NotRequired[bool]
    execution_mode: NotRequired[str]
    answer_follow_ups: NotRequired[list[str]]
    answer_charts: NotRequired[list[dict[str, object]]]

    active_task_ids: NotRequired[list[str]]
    replan_count: NotRequired[int]
    next_action: NotRequired[str]
    execution_status: NotRequired[str]
    summary: NotRequired[str]
    route: NotRequired[str]
    execution_lane: NotRequired[str]
    guardrail_decision: NotRequired[dict[str, object]]
    guardrails_pass: NotRequired[bool]
    guardrails_reason: NotRequired[str]
    post_turn_memory_status: NotRequired[str]
    post_turn_memory_enqueued: NotRequired[list[str]]


__all__ = ["OrchestratorState"]
