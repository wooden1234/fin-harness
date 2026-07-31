"""Analyzer 语义输出校验与确定性执行解析。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.analyzer.heuristic import (
    market_tool_for_query,
    research_tool_for_query,
)
from agents.orchestrator.analyzer.schema import AnalyzerOutput
from agents.orchestrator.contracts import (
    ExecutionDecision,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)

_MAX_ENTITIES = 8
_RESEARCH_INTENTS = {"open_research", "entity_comparison"}
_CORPORATE_POLICY_MARKERS = (
    "报销",
    "发票",
    "付款",
    "审批",
    "预算",
    "内控",
    "备用金",
    "关账",
    "盘点",
    "财务制度",
)


def _corporate_policy_explicit(query: str) -> bool:
    return any(marker in query for marker in _CORPORATE_POLICY_MARKERS)


@dataclass(frozen=True, slots=True)
class ProfileValidation:
    profile: RequestProfile
    issues: list[str]
    needs_repair: bool


def _execution_decision(
    *,
    intents: list[str],
    query: str,
    missing_fields: list[str],
    constraints: dict,
) -> ExecutionDecision:
    intent_set = set(intents)
    if missing_fields or "clarify" in intent_set:
        return ExecutionDecision(mode="clarify", budget_tier="light")
    if intent_set & _RESEARCH_INTENTS:
        sources = ["market", "research", "finance_rag", "local_documents"]
        if intent_set & {"concept_explain", "product_policy"}:
            sources.append("stable_rules")
        knowledge_scope = ["approved:pdf"]
        if _corporate_policy_explicit(query):
            knowledge_scope.append("explicit:corporate_finance")
        return ExecutionDecision(
            mode="deep_research",
            budget_tier="research",
            allowed_capabilities=["deep.research"],
            data_sources=sources,
            knowledge_scope=knowledge_scope,
        )
    if "candidate_compute" in intent_set:
        return ExecutionDecision(
            mode="market_compute",
            budget_tier="standard",
            allowed_capabilities=["market.compute"],
            data_sources=["upstream_data"],
        )
    if "stock_screening" in intent_set:
        return ExecutionDecision(
            mode="stock_screen",
            budget_tier="standard",
            allowed_capabilities=["iwencai.screen"],
            data_sources=["market"],
        )
    if "research_search" in intent_set:
        tool_id = research_tool_for_query(query)
        return ExecutionDecision(
            mode="research_retrieve",
            budget_tier="standard",
            allowed_capabilities=[tool_id] if tool_id else [],
            data_sources=["research"],
            tool_id=tool_id,
        )
    if "market_query" in intent_set:
        tool_id = market_tool_for_query(query)
        return ExecutionDecision(
            mode="market_acquire",
            budget_tier="standard",
            allowed_capabilities=[tool_id] if tool_id else [],
            data_sources=["market"],
            tool_id=tool_id,
        )
    if "general_chat" in intent_set:
        return ExecutionDecision(mode="general_answer", budget_tier="light")
    if "document_qa" in intent_set:
        document = constraints.get("document") or {}
        categories = list(document.get("categories") or []) if isinstance(document, dict) else []
        return ExecutionDecision(
            mode="document_qa",
            budget_tier="standard",
            allowed_capabilities=["pdf"],
            data_sources=["finance_rag"],
            knowledge_scope=categories or ["approved:pdf"],
        )
    if "structured_metric" in intent_set:
        return ExecutionDecision(
            mode="structured_finance",
            budget_tier="standard",
            allowed_capabilities=["financial_query"],
            data_sources=["finance_rag"],
        )
    return ExecutionDecision(
        mode="faq_lookup",
        budget_tier="standard",
        allowed_capabilities=["faq"],
        data_sources=["finance_rag"],
        knowledge_scope=[
            "explicit:corporate_finance"
            if "product_policy" in intent_set and _corporate_policy_explicit(query)
            else "capital_market"
        ],
    )


def validate_and_normalize(
    raw: AnalyzerOutput,
    *,
    original_query: str,
) -> ProfileValidation:
    """校验语义输出，并由本地规则生成唯一执行决策。"""
    hard_issues: list[str] = []
    query = (raw.normalized_query or original_query or "").strip()
    intents = list(dict.fromkeys(raw.intents))
    missing_fields = list(dict.fromkeys(item for item in raw.missing_fields if item))
    clarification_message = raw.clarification_message.strip()
    constraints = raw.constraints.model_dump(mode="json", exclude_none=True)

    if not query and "query" not in missing_fields:
        missing_fields.append("query")
    if not missing_fields and not intents:
        hard_issues.append("empty_intents")
    if missing_fields and "clarify" not in intents:
        intents.append("clarify")
    if not missing_fields:
        clarification_message = ""

    entities = list(
        dict.fromkeys(item.strip() for item in raw.entities if item and item.strip())
    )[:_MAX_ENTITIES]
    execution = _execution_decision(
        intents=intents,
        query=query,
        missing_fields=missing_fields,
        constraints=constraints,
    )
    if execution.mode == "market_compute" and not (
        constraints.get("candidate_set_id") or constraints.get("market_query_plan")
    ):
        hard_issues.append("candidate_compute_requires_artifact_or_plan")
    if execution.mode in {"market_acquire", "research_retrieve"} and not execution.tool_id:
        hard_issues.append(f"{execution.mode}_tool_unresolved")

    profile = RequestProfile(
        original_query=original_query,
        normalized_query=query or original_query,
        intents=intents,
        freshness_required=bool(raw.freshness_required),
        entities=entities,
        constraints=constraints,
        missing_fields=missing_fields,
        clarification_message=clarification_message,
        execution=execution,
    )
    return ProfileValidation(
        profile=profile,
        issues=hard_issues,
        needs_repair=bool(hard_issues),
    )


def assert_task_capabilities(task: TaskSpec) -> None:
    spec = get_agent_spec(task.agent_id)
    missing = [item for item in task.required_capabilities if item not in spec.capabilities]
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
