"""独立 Research Workflow 的边界与组合测试。"""

from __future__ import annotations

import asyncio

from agents.orchestrator.contracts import AgentResult, Evidence
from agents.research_workflow import (
    build_research_plan,
    build_research_workflow,
    plan_research_adaptive,
    run_research_workflow,
)
from agents.research_workflow.contracts import ResearchPlanDraft


class _FakeStructuredModel:
    def __init__(self, output=None, error: BaseException | None = None):
        self.output = output
        self.error = error

    def with_structured_output(self, schema, method=None):
        del schema, method
        return self

    async def ainvoke(self, messages, config=None):
        del messages, config
        if self.error is not None:
            raise self.error
        return self.output


def test_research_workflow_compiles_with_deep_agent_as_internal_node() -> None:
    graph = build_research_workflow().compile().get_graph()

    assert "plan_research" in graph.nodes
    assert "collect_sources" in graph.nodes
    assert "deep_research" in graph.nodes
    assert "finalize_research" in graph.nodes


def test_research_planner_owns_internal_source_tasks() -> None:
    plan = build_research_plan(
        "研究测试公司",
        {
            "data_sources": ["research", "finance_rag"],
            "entities": ["测试公司"],
        },
    )

    assert plan.entities == ["测试公司"]
    assert [task.task_id for task in plan.source_tasks] == [
        "research:announcement",
        "research:report",
        "research:finance",
    ]
    assert plan.questions[-1].question_id == "critical_review"


async def test_research_planner_uses_validated_llm_plan() -> None:
    draft = ResearchPlanDraft(
        data_sources=["research"],
        questions=[
            {
                "question_id": "event_impact",
                "objective": "近期公告是否改变核心研究假设？",
                "evidence_requirements": ["公告时间和摘要"],
            }
        ],
        rationale="优先核验近期公开披露。",
    )

    plan = await plan_research_adaptive(
        "研究测试公司近期风险",
        {"data_sources": ["research"], "entities": ["测试公司"]},
        llm=_FakeStructuredModel(draft),
    )

    assert plan.metadata["planner"] == "llm_validated"
    assert plan.data_sources == ["research"]
    assert [item.question_id for item in plan.questions] == [
        "event_impact",
        "critical_review",
    ]
    assert [task.task_id for task in plan.source_tasks] == [
        "research:announcement",
        "research:report",
    ]


async def test_research_planner_rejects_sources_outside_root_scope() -> None:
    draft = ResearchPlanDraft(
        data_sources=["market", "research", "web"],
        questions=[
            {
                "question_id": "evidence",
                "objective": "有哪些公开证据？",
            }
        ],
    )

    plan = await plan_research_adaptive(
        "研究测试公司",
        {"data_sources": ["research"]},
        llm=_FakeStructuredModel(draft),
    )

    assert plan.data_sources == ["research"]
    assert any(
        item.startswith("unsupported_data_sources:")
        for item in plan.metadata["validation_issues"]
    )


async def test_research_planner_falls_back_when_llm_fails() -> None:
    plan = await plan_research_adaptive(
        "研究测试公司",
        {"data_sources": ["research"]},
        llm=_FakeStructuredModel(error=TimeoutError("boom")),
    )

    assert plan.metadata["planner"] == "deterministic_fallback"
    assert plan.metadata["fallback_reason"] == "TimeoutError"
    assert plan.questions[-1].question_id == "critical_review"


async def test_research_workflow_returns_single_result(monkeypatch) -> None:
    async def fake_plan(query, task_input, *, config=None):
        del config
        return build_research_plan(query, task_input)

    async def fake_invoke_agent(
        task,
        *,
        dependency_results,
        config=None,
        runtime=None,
    ):
        del dependency_results, config, runtime
        return AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="completed",
            answer=f"{task.task_id} 已完成",
            evidence=[
                Evidence(
                    evidence_id=f"evidence:{task.task_id}",
                    task_id=task.task_id,
                    source_type="iwencai",
                )
            ],
        )

    async def fake_deep_research(
        state,
        *,
        query,
        config=None,
        runtime=None,
        **kwargs,
    ):
        del config, runtime, kwargs
        dependencies = list(state.get("dependency_results") or [])
        assert dependencies[0].agent_id == "research_workflow.planner"
        return AgentResult(
            task_id="deep-research",
            agent_id="deep_research_agent",
            status="completed",
            answer=f"完成研究：{query}",
            evidence=[
                item
                for dependency in dependencies
                for item in dependency.evidence
            ],
        )

    monkeypatch.setattr(
        "agents.research_workflow.workflow.plan_research_adaptive",
        fake_plan,
    )
    monkeypatch.setattr(
        "agents.orchestrator.agent_registry.invoke_agent",
        fake_invoke_agent,
    )
    monkeypatch.setattr(
        "agents.research_workflow.deep_agent.run_deep_research_agent",
        fake_deep_research,
    )

    result = await run_research_workflow(
        {
            "messages": [],
            "task_input": {"data_sources": ["research"]},
        },
        query="研究测试公司",
    )

    assert result.agent_id == "research_workflow"
    assert result.status == "completed"
    assert result.answer == "完成研究：研究测试公司"
    assert result.metadata["deep_agent_id"] == "deep_research_agent"
    assert result.metadata["research_plan"]["query"] == "研究测试公司"
    assert result.metadata["source_task_ids"] == [
        "research:announcement",
        "research:report",
    ]
    assert len(result.evidence) == 2


async def test_research_source_collection_limits_concurrency(
    monkeypatch,
) -> None:
    from agents.research_workflow.workflow import collect_research_sources

    active = 0
    max_active = 0

    async def fake_invoke_agent(task, **kwargs):
        nonlocal active, max_active
        del kwargs
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="completed",
        )

    monkeypatch.setattr(
        "agents.orchestrator.agent_registry.invoke_agent",
        fake_invoke_agent,
    )
    monkeypatch.setattr(
        "agents.research_workflow.workflow.settings.AGENT_V2_MAX_CONCURRENCY",
        2,
    )
    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["market", "research", "finance_rag"]},
    )

    result = await collect_research_sources({"research_plan": plan})

    assert len(result["source_results"]) == 4
    assert max_active == 2
