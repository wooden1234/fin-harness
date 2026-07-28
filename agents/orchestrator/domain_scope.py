"""领域 Planner 授权范围的构造与确定性校验。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.contracts import (
    DomainPlanningScope,
    EvidencePolicy,
    TaskSpec,
)


_FINANCE_INTENTS_BY_CAPABILITY = {
    "faq": ("concept_explain", "product_policy"),
    "pdf": ("document_qa",),
    "financial_query": ("structured_metric",),
    "web_search": ("market_event",),
}


def _allowed_intents(capabilities: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(
            intent
            for capability in capabilities
            for intent in _FINANCE_INTENTS_BY_CAPABILITY.get(capability, ())
        )
    )


def build_finance_scope(
    *,
    parent_task_id: str,
    parent_logical_task_id: str,
    data_sources: Iterable[str] = (),
    freshness_required: bool = False,
    entities: Iterable[str] = (),
    constraints: Mapping[str, Any] | None = None,
    evidence_policy: EvidencePolicy | None = None,
) -> DomainPlanningScope:
    """按注册表能力生成 Finance Planner 的授权范围。"""
    spec = get_agent_spec("finance_agent")
    capabilities = list(spec.capabilities)
    return DomainPlanningScope(
        parent_task_id=parent_task_id,
        parent_logical_task_id=parent_logical_task_id,
        allowed_capabilities=capabilities,
        allowed_data_sources=list(data_sources),
        allowed_intents=_allowed_intents(capabilities),
        fallback_policy="within_scope",
        freshness_required=freshness_required,
        evidence_policy=evidence_policy or spec.evidence_policy,
        entities=list(entities),
        constraints=dict(constraints or {}),
        max_subtasks=4,
    )


def ensure_task_domain_scope(task: TaskSpec) -> TaskSpec:
    """为 V2 Finance 任务补齐 Scope，并拒绝注册能力之外的授权。"""
    if task.agent_id != "finance_agent":
        return task

    raw_scope = task.input_data.get("domain_planning_scope")
    if raw_scope is None:
        scope = build_finance_scope(
            parent_task_id=task.task_id,
            parent_logical_task_id=task.logical_task_id or task.task_id,
        )
    else:
        scope = DomainPlanningScope.model_validate(raw_scope)

    registered = set(get_agent_spec("finance_agent").capabilities)
    extra_capabilities = sorted(
        set(scope.allowed_capabilities) - registered
    )
    if extra_capabilities:
        raise ValueError(
            "domain_scope_capability_mismatch:finance_agent:"
            + ",".join(extra_capabilities)
        )

    capability_intents = set(_allowed_intents(scope.allowed_capabilities))
    extra_intents = sorted(set(scope.allowed_intents) - capability_intents)
    if extra_intents:
        raise ValueError(
            "domain_scope_intent_mismatch:finance_agent:"
            + ",".join(extra_intents)
        )

    input_data = {
        **task.input_data,
        "domain_planning_scope": scope.model_dump(mode="json"),
    }
    return task.model_copy(update={"input_data": input_data})


__all__ = ["build_finance_scope", "ensure_task_domain_scope"]
