"""根据 RequestProfile 确定性生成 TaskPlan。"""

from __future__ import annotations

import uuid

from agents.orchestrator.analyzer.validate import assert_plan_capabilities
from agents.orchestrator.capability_resolver import resolve_finance_capabilities
from agents.orchestrator.contracts import (
    RequestProfile,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.domain_scope import build_finance_scope


def _finance_planning_scope(profile: RequestProfile):
    """把请求画像转换为 Finance Planner 的显式授权范围。"""
    return build_finance_scope(
        parent_task_id="finance",
        parent_logical_task_id="finance",
        data_sources=profile.execution.data_sources,
        freshness_required=profile.freshness_required,
        entities=profile.entities,
        constraints=profile.constraints,
    )


def _needs_semantic_history(profile: RequestProfile) -> bool:
    if profile.constraints.get("semantic_history") is True:
        return True
    query = profile.normalized_query or profile.original_query
    markers = (
        "我之前",
        "我过去",
        "我们之前",
        "上次我们",
        "之前聊过",
        "根据我的历史",
        "结合我的历史",
        "回顾我的",
    )
    return any(marker in query for marker in markers)


def build_plan_from_profile(profile: RequestProfile) -> TaskPlan:
    """规则模板编计划；不调用 LLM。"""
    plan_id = f"plan-{uuid.uuid4().hex[:12]}"
    query = profile.normalized_query or profile.original_query
    semantic_history = _needs_semantic_history(profile)
    execution = profile.execution
    if profile.missing_fields or execution.mode == "clarify":
        return TaskPlan(plan_id=plan_id, query=query, tasks=[])

    if execution.mode in {
        "market_acquire",
        "research_retrieve",
        "stock_screen",
        "market_compute",
        "deep_research",
    }:
        # 独立研究工作流已下线；主路径由 execution_lane → main_deep_agent 承接。
        return TaskPlan(
            plan_id=plan_id,
            query=query,
            tasks=[],
            metadata={
                "planning_status": "uncovered",
                "reason": "legacy_research_workflow_retired",
            },
        )

    if execution.mode == "general_answer":
        plan = TaskPlan(
            plan_id=plan_id,
            query=query,
            tasks=[
                TaskSpec(
                    task_id="general",
                    objective=query,
                    agent_id="general_agent",
                    required_capabilities=[],
                    semantic_history=semantic_history,
                )
            ],
        )
        assert_plan_capabilities(plan)
        return plan

    finance_scope = _finance_planning_scope(profile)
    required_capabilities = resolve_finance_capabilities(
        profile,
        allowed_capabilities=finance_scope.allowed_capabilities,
    )
    if not required_capabilities:
        return TaskPlan(
            plan_id=plan_id,
            query=query,
            tasks=[],
            metadata={
                "planning_status": "uncovered",
                "error_code": "no_finance_capability_in_scope",
            },
        )
    plan = TaskPlan(
        plan_id=plan_id,
        query=query,
        tasks=[
            TaskSpec(
                task_id="finance",
                objective=query,
                agent_id="finance_agent",
                required_capabilities=required_capabilities,
                semantic_history=semantic_history,
                input_data={
                    "domain_planning_scope": finance_scope.model_dump(
                        mode="json"
                    ),
                    # Root 统一负责成稿，Finance 仅返回结构化证据，避免重复 LLM 总结。
                    "output_mode": "evidence_only",
                },
            )
        ],
    )
    assert_plan_capabilities(plan)
    return plan


__all__ = ["build_plan_from_profile"]
