"""顶层编排器的公共数据契约。"""

from agents.orchestrator.contracts import (
    AgentResult,
    DomainPlanningScope,
    ErrorAction,
    Evidence,
    EvidencePolicy,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.error_policy import ErrorDecision, classify_error
from agents.orchestrator.domain_scope import (
    build_finance_scope,
    ensure_task_domain_scope,
)
from agents.orchestrator.task_identity import (
    build_idempotency_key,
    ensure_task_identity,
    validate_task_plan,
)
from agents.orchestrator.graph import build_orchestrator_graph, get_orchestrator_graph

__all__ = [
    "AgentResult",
    "DomainPlanningScope",
    "ErrorAction",
    "ErrorDecision",
    "Evidence",
    "EvidencePolicy",
    "classify_error",
    "build_finance_scope",
    "ensure_task_domain_scope",
    "build_idempotency_key",
    "ensure_task_identity",
    "validate_task_plan",
    "RequestProfile",
    "TaskPlan",
    "TaskSpec",
    "build_orchestrator_graph",
    "get_orchestrator_graph",
]
