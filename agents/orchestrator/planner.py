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

    if profile.preferred_agent == "stock_screening_agent":
        screen = TaskSpec(
            task_id="screen",
            objective=query,
            agent_id="stock_screening_agent",
            required_capabilities=["iwencai.screen"],
        )
        if profile.complexity == "compound":
            research = TaskSpec(
                task_id="research",
                objective="根据上游选股结果，分析候选股票的财务表现、公开信息和主要风险",
                agent_id="finance_agent",
                depends_on=["screen"],
                required_capabilities=["financial_query", "web_search"],
            )
            plan = TaskPlan(plan_id=plan_id, query=query, tasks=[screen, research])
            assert_plan_capabilities(plan)
            return plan
        plan = TaskPlan(plan_id=plan_id, query=query, tasks=[screen])
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
                required_capabilities=["financial_query", "web_search"],
            )
        ],
    )
    assert_plan_capabilities(plan)
    return plan


__all__ = ["build_plan_from_profile"]
