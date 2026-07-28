"""根据 RequestProfile 确定性生成 TaskPlan。"""

from __future__ import annotations

import uuid

from agents.orchestrator.analyzer.validate import assert_plan_capabilities
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
        data_sources=profile.data_sources,
        freshness_required=profile.freshness_required,
        entities=profile.entities,
        constraints=profile.constraints,
    )


def build_plan_from_profile(profile: RequestProfile) -> TaskPlan:
    """规则模板编计划；不调用 LLM。"""
    plan_id = f"plan-{uuid.uuid4().hex[:12]}"
    query = profile.normalized_query or profile.original_query
    if profile.missing_fields:
        return TaskPlan(plan_id=plan_id, query=query, tasks=[])

    if profile.preferred_agent == "market_acquisition_workflow":
        market_tool_id = str(profile.constraints.get("market_tool_id") or "").strip()
        acquire = TaskSpec(
            task_id="market_acquire",
            objective=query,
            agent_id="market_acquisition_workflow",
            required_capabilities=[market_tool_id] if market_tool_id else [],
            input_data={"market_tool_id": market_tool_id} if market_tool_id else {},
        )
        plan = TaskPlan(plan_id=plan_id, query=query, tasks=[acquire])
        assert_plan_capabilities(plan)
        return plan

    if profile.preferred_agent == "research_retrieval_workflow":
        research_tool_id = str(
            profile.constraints.get("research_tool_id") or ""
        ).strip()
        research = TaskSpec(
            task_id="research",
            objective=query,
            agent_id="research_retrieval_workflow",
            required_capabilities=[research_tool_id] if research_tool_id else [],
            input_data=(
                {"research_tool_id": research_tool_id}
                if research_tool_id
                else {}
            ),
        )
        plan = TaskPlan(plan_id=plan_id, query=query, tasks=[research])
        assert_plan_capabilities(plan)
        return plan

    if profile.preferred_agent == "stock_screening_agent":
        screening = TaskSpec(
            task_id="stock_screening",
            objective=query,
            agent_id="stock_screening_agent",
            required_capabilities=["iwencai.screen"],
        )
        plan = TaskPlan(plan_id=plan_id, query=query, tasks=[screening])
        assert_plan_capabilities(plan)
        return plan

    if profile.preferred_agent == "market.compute":
        query_plan = profile.constraints.get("market_query_plan")
        compute = TaskSpec(
            task_id="market_compute",
            objective=query,
            agent_id="market.compute",
            required_capabilities=["market.compute"],
            input_data=(
                {"market_query_plan": query_plan}
                if query_plan is not None
                else {}
            ),
        )
        plan = TaskPlan(plan_id=plan_id, query=query, tasks=[compute])
        assert_plan_capabilities(plan)
        return plan

    if profile.preferred_agent in {"deep_research_agent", "research_workflow"}:
        research = TaskSpec(
            task_id="research",
            objective=query,
            agent_id="research_workflow",
            required_capabilities=["deep.research"],
            input_data={
                "data_sources": list(profile.data_sources),
                "entities": list(profile.entities),
            },
        )
        plan = TaskPlan(
            plan_id=plan_id,
            query=query,
            tasks=[research],
        )
        assert_plan_capabilities(plan)
        return plan

    if profile.preferred_agent == "general_agent":
        plan = TaskPlan(
            plan_id=plan_id,
            query=query,
            tasks=[
                TaskSpec(
                    task_id="general",
                    objective=query,
                    agent_id="general_agent",
                    required_capabilities=[],
                )
            ],
        )
        assert_plan_capabilities(plan)
        return plan

    finance_scope = _finance_planning_scope(profile)
    plan = TaskPlan(
        plan_id=plan_id,
        query=query,
        tasks=[
            TaskSpec(
                task_id="finance",
                objective=query,
                agent_id="finance_agent",
                required_capabilities=["financial_query"],
                input_data={
                    "domain_planning_scope": finance_scope.model_dump(
                        mode="json"
                    )
                },
            )
        ],
    )
    assert_plan_capabilities(plan)
    return plan


__all__ = ["build_plan_from_profile"]
