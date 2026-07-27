"""根据 RequestProfile 确定性生成 TaskPlan。"""

from __future__ import annotations

import uuid

from agents.orchestrator.analyzer.validate import assert_plan_capabilities
from agents.orchestrator.contracts import RequestProfile, TaskPlan, TaskSpec


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

    plan = TaskPlan(
        plan_id=plan_id,
        query=query,
        tasks=[
            TaskSpec(
                task_id="finance",
                objective=query,
                agent_id="finance_agent",
                required_capabilities=["financial_query"],
            )
        ],
    )
    assert_plan_capabilities(plan)
    return plan


__all__ = ["build_plan_from_profile"]
