"""请求画像与任务能力的确定性校验。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.orchestrator.agent_registry import get_agent_spec, list_agent_specs
from agents.orchestrator.analyzer.heuristic import (
    market_tool_for_query,
    research_tool_for_query,
)
from agents.orchestrator.analyzer.schema import AnalyzerOutput
from agents.orchestrator.contracts import RequestProfile, TaskPlan, TaskSpec

_REGISTERED_AGENTS = {item.agent_id for item in list_agent_specs()}
_MAX_ENTITIES = 8


@dataclass(frozen=True, slots=True)
class ProfileValidation:
    profile: RequestProfile
    issues: list[str]
    needs_repair: bool


def validate_and_normalize(
    raw: AnalyzerOutput,
    *,
    original_query: str,
) -> ProfileValidation:
    """将 LLM 输出收敛为合法 RequestProfile，并标记是否需要 repair。"""
    issues: list[str] = []
    hard_issues: list[str] = []
    query = (raw.normalized_query or original_query or "").strip()
    intents = list(dict.fromkeys(raw.intents))
    missing_fields = [item for item in raw.missing_fields if item]
    preferred = raw.preferred_agent
    complexity = raw.complexity
    data_sources = list(dict.fromkeys(raw.data_sources))
    operation_type = raw.operation_type
    constraints = dict(raw.constraints or {})

    if not query and "query" not in missing_fields:
        missing_fields.append("query")

    if preferred is not None and preferred not in _REGISTERED_AGENTS:
        issues.append(f"unknown_preferred_agent:{preferred}")
        preferred = None

    if missing_fields:
        preferred = None
        complexity = "simple"
        data_sources = []
        operation_type = "answer"
        if not intents:
            intents = ["clarify"]
    elif not intents:
        hard_issues.append("empty_intents")
    elif "deep_research" in intents or (
        "stock_screening" in intents and "financial_analysis" in intents
    ):
        preferred = "research_workflow"
        complexity = "compound"
        data_sources = data_sources or ["market", "research", "finance_rag"]
        operation_type = "deep_research"
    elif "market_compute" in intents:
        preferred = "market.compute"
        data_sources = data_sources or ["upstream_data"]
        operation_type = "compute"
    elif "stock_screening" in intents:
        if preferred not in {None, "stock_screening_agent"}:
            issues.append("stock_screening_requires_stock_screening_agent")
        preferred = "stock_screening_agent"
        data_sources = data_sources or ["market"]
        operation_type = "acquire"
    elif "research_search" in intents or research_tool_for_query(query):
        preferred = "research_retrieval_workflow"
        data_sources = data_sources or ["research"]
        operation_type = "retrieve"
        research_tool_id = research_tool_for_query(query)
        if research_tool_id:
            constraints.setdefault("research_tool_id", research_tool_id)
    elif "market_query" in intents or market_tool_for_query(query):
        preferred = "market_acquisition_workflow"
        data_sources = data_sources or ["market"]
        operation_type = "acquire"
        market_tool_id = market_tool_for_query(query)
        if market_tool_id:
            constraints.setdefault("market_tool_id", market_tool_id)
    elif "general_chat" in intents and not any(
        item in intents
        for item in (
            "financial_analysis",
            "financial_research",
            "stock_screening",
            "deep_research",
        )
    ):
        preferred = preferred or "general_agent"
        data_sources = data_sources or ["none"]
        operation_type = "answer"
    else:
        if preferred not in {None, "finance_agent"}:
            issues.append("finance_query_routed_to_wrong_agent")
        preferred = "finance_agent"
        data_sources = data_sources or ["finance_rag"]
        operation_type = (
            operation_type
            if operation_type in {"retrieve", "analyze"}
            else "analyze"
        )

    if (
        complexity == "compound"
        and len(intents) < 2
        and "deep_research" not in intents
    ):
        issues.append("compound_requires_multiple_intents")
        complexity = "single_capability"

    if not missing_fields and preferred is None and not hard_issues:
        hard_issues.append("unresolved_preferred_agent")

    entities = [item.strip() for item in raw.entities if item and item.strip()]
    entities = list(dict.fromkeys(entities))[:_MAX_ENTITIES]

    if preferred == "market_acquisition_workflow":
        market_tool_id = market_tool_for_query(query)
        if market_tool_id:
            constraints.setdefault("market_tool_id", market_tool_id)
    elif preferred == "research_retrieval_workflow":
        research_tool_id = research_tool_for_query(query)
        if research_tool_id:
            constraints.setdefault("research_tool_id", research_tool_id)

    profile = RequestProfile(
        original_query=original_query,
        normalized_query=query or original_query,
        intents=intents,
        complexity=complexity,
        data_sources=data_sources,
        operation_type=operation_type,
        freshness_required=bool(raw.freshness_required),
        entities=entities,
        constraints=constraints,
        missing_fields=missing_fields,
        preferred_agent=preferred,
    )
    all_issues = issues + hard_issues
    return ProfileValidation(
        profile=profile,
        issues=all_issues,
        needs_repair=bool(hard_issues),
    )


def assert_task_capabilities(task: TaskSpec) -> None:
    """校验任务声明的能力是目标 Agent 的子集。"""
    spec = get_agent_spec(task.agent_id)
    missing = [
        item for item in task.required_capabilities if item not in spec.capabilities
    ]
    if missing:
        raise ValueError(
            f"capability_mismatch:{task.agent_id}:{','.join(sorted(missing))}"
        )


def assert_plan_capabilities(plan: TaskPlan) -> None:
    for task in plan.tasks:
        assert_task_capabilities(task)


__all__ = [
    "ProfileValidation",
    "assert_plan_capabilities",
    "assert_task_capabilities",
    "validate_and_normalize",
]
