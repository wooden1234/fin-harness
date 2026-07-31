"""独立 Research Workflow 的边界与组合测试。"""

from __future__ import annotations

import asyncio

import pytest

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
    assert "validate_question_evidence" in graph.nodes
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
    assert all(task.logical_task_id for task in plan.source_tasks)
    assert all(task.attempt_id for task in plan.source_tasks)
    assert all(task.idempotency_key for task in plan.source_tasks)
    assert all(task.evidence_policy is not None for task in plan.source_tasks)
    finance_task = next(
        task for task in plan.source_tasks if task.task_id == "research:finance"
    )
    assert "domain_planning_scope" in finance_task.input_data
    public_disclosures = next(
        item
        for item in plan.questions
        if item.question_id == "public_disclosures"
    )
    assert public_disclosures.source_task_ids == ["research:announcement"]
    assert public_disclosures.evidence_policy.required is True
    assert public_disclosures.evidence_policy.require_provenance is True
    critical_review = plan.questions[-1]
    assert critical_review.source_task_ids == [
        "research:announcement",
        "research:report",
        "research:finance",
    ]
    assert critical_review.evidence_policy.min_count == 2


def test_research_planner_distinguishes_missing_and_explicit_empty_scope() -> None:
    defaulted = build_research_plan("研究测试公司", {})
    blocked = build_research_plan(
        "研究测试公司",
        {"data_sources": []},
    )

    assert defaulted.data_sources == [
        "market",
        "research",
        "finance_rag",
        "local_documents",
    ]
    assert defaulted.metadata["source_scope_origin"] == "missing_default"
    assert blocked.data_sources == []
    assert blocked.source_tasks == []
    assert blocked.questions == []
    assert blocked.metadata["source_scope_origin"] == "explicit_empty"
    assert blocked.metadata["scope_blocked"] is True


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
    event_impact = plan.questions[0]
    assert event_impact.source_task_ids == [
        "research:announcement",
        "research:report",
    ]
    assert event_impact.evidence_policy.min_count == 1


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


async def test_research_planner_explicit_empty_scope_skips_llm() -> None:
    plan = await plan_research_adaptive(
        "研究测试公司",
        {"data_sources": []},
        llm=_FakeStructuredModel(error=AssertionError("不应调用 LLM")),
    )

    assert plan.metadata["planner"] == "scope_blocked"
    assert plan.source_tasks == []


async def test_research_planner_does_not_expand_explicit_empty_llm_scope() -> None:
    draft = ResearchPlanDraft(
        data_sources=[],
        questions=[
            {
                "question_id": "event_impact",
                "objective": "公告是否改变研究假设？",
            }
        ],
    )

    plan = await plan_research_adaptive(
        "研究测试公司",
        {"data_sources": ["research"]},
        llm=_FakeStructuredModel(draft),
    )

    assert plan.data_sources == []
    assert plan.source_tasks == []
    assert plan.questions == []
    assert plan.metadata["scope_blocked"] is True
    assert "llm_data_sources_explicit_empty" in plan.metadata[
        "validation_issues"
    ]


async def test_research_workflow_returns_single_result(monkeypatch) -> None:
    async def fake_plan(
        query,
        task_input,
        *,
        task_identity=None,
        config=None,
    ):
        del config
        return build_research_plan(
            query,
            task_input,
            task_identity=task_identity,
        )

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
                    provider="iwencai",
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
    assert all(
        item["passed"]
        for item in result.metadata["question_evidence_assessments"]
    )
    assert len(result.evidence) == 2


async def test_research_workflow_explicit_empty_scope_returns_uncovered() -> None:
    result = await run_research_workflow(
        {
            "messages": [],
            "task_input": {"data_sources": []},
        },
        query="研究测试公司",
    )

    assert result.status == "uncovered"
    assert result.error_code == "research_scope_empty"
    assert result.metadata["source_task_ids"] == []


async def test_research_source_tasks_are_revalidated_before_execution() -> None:
    from agents.research_workflow.workflow import collect_research_sources

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["research"]},
    )
    tampered_task = plan.source_tasks[0].model_copy(
        update={"idempotency_key": "tampered"}
    )
    tampered_plan = plan.model_copy(
        update={"source_tasks": [tampered_task, *plan.source_tasks[1:]]}
    )

    with pytest.raises(ValueError, match="idempotency_key_mismatch"):
        await collect_research_sources({"research_plan": tampered_plan})


async def test_research_source_collection_normalizes_evidence_task_id(
    monkeypatch,
) -> None:
    from agents.research_workflow.workflow import collect_research_sources

    async def fake_invoke_agent(task, **kwargs):
        del kwargs
        return AgentResult(
            task_id="workflow-default-task",
            agent_id=task.agent_id,
            status="completed",
            evidence=[
                Evidence(
                    evidence_id="source-evidence",
                    task_id="workflow-default-task",
                    source_type="iwencai",
                    provider="iwencai",
                )
            ],
        )

    monkeypatch.setattr(
        "agents.orchestrator.agent_registry.invoke_agent",
        fake_invoke_agent,
    )
    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["research"]},
    )

    update = await collect_research_sources({"research_plan": plan})

    assert [
        result.evidence[0].task_id
        for result in update["source_results"]
    ] == ["research:announcement", "research:report"]


async def test_research_source_task_capability_expansion_is_rejected() -> None:
    from agents.research_workflow.workflow import collect_research_sources

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["research"]},
    )
    expanded_task = plan.source_tasks[0].model_copy(
        update={
            "required_capabilities": ["iwencai.screen"],
            "idempotency_key": "",
        }
    )
    expanded_plan = plan.model_copy(
        update={"source_tasks": [expanded_task, *plan.source_tasks[1:]]}
    )

    with pytest.raises(ValueError, match="capability_mismatch"):
        await collect_research_sources({"research_plan": expanded_plan})


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


async def test_question_evidence_does_not_cross_source_task_boundary() -> None:
    from agents.research_workflow.workflow import validate_question_evidence

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["research"]},
    )
    source_results = [
        AgentResult(
            task_id="research:announcement",
            agent_id="research_retrieval_workflow",
            status="completed",
            evidence=[
                Evidence(
                    evidence_id="misattributed-report",
                    task_id="research:report",
                    source_type="iwencai.report.search",
                    provider="iwencai",
                )
            ],
        ),
        AgentResult(
            task_id="research:report",
            agent_id="research_retrieval_workflow",
            status="completed",
            evidence=[
                Evidence(
                    evidence_id="report-evidence",
                    task_id="research:report",
                    source_type="iwencai.report.search",
                    provider="iwencai",
                )
            ],
        ),
    ]

    update = await validate_question_evidence(
        {
            "research_plan": plan,
            "source_results": source_results,
        }
    )
    assessments = {
        item.question_id: item
        for item in update["question_evidence_assessments"]
    }

    assert assessments["public_disclosures"].passed is False
    assert assessments["public_disclosures"].evidence_count == 0
    assert assessments["institution_views"].passed is True
    assert assessments["institution_views"].evidence_ids == [
        "report-evidence"
    ]


async def test_question_evidence_requires_market_structured_data() -> None:
    from agents.research_workflow.workflow import validate_question_evidence

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["market"]},
    )
    update = await validate_question_evidence(
        {
            "research_plan": plan,
            "source_results": [
                AgentResult(
                    task_id="research:stock_screening",
                    agent_id="stock_screening_agent",
                    status="completed",
                    evidence=[
                        Evidence(
                            evidence_id="market-evidence",
                            task_id="research:stock_screening",
                            source_type="iwencai",
                            provider="iwencai",
                        )
                    ],
                )
            ],
        }
    )
    assessments = {
        item.question_id: item
        for item in update["question_evidence_assessments"]
    }

    market = assessments["market_context"]
    assert market.passed is False
    assert market.structured_data_present is False
    assert market.gaps == ["structured_data_missing"]


async def test_question_evidence_requires_provenance() -> None:
    from agents.research_workflow.workflow import validate_question_evidence

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["finance_rag"]},
    )
    update = await validate_question_evidence(
        {
            "research_plan": plan,
            "source_results": [
                AgentResult(
                    task_id="research:finance",
                    agent_id="finance_agent",
                    status="completed",
                    evidence=[
                        Evidence(
                            evidence_id="finance-without-provenance",
                            task_id="research:finance",
                            source_type="financial_db",
                        )
                    ],
                )
            ],
        }
    )
    assessments = {
        item.question_id: item
        for item in update["question_evidence_assessments"]
    }

    financial = assessments["financial_facts"]
    assert financial.passed is False
    assert financial.missing_provenance_count == 1
    assert financial.gaps == ["evidence_provenance_missing:1"]


async def test_finalize_research_downgrades_partial_question_coverage() -> None:
    from agents.research_workflow.workflow import (
        finalize_research,
        validate_question_evidence,
    )

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["research"]},
    )
    source_results = [
        AgentResult(
            task_id="research:announcement",
            agent_id="research_retrieval_workflow",
            status="completed",
        ),
        AgentResult(
            task_id="research:report",
            agent_id="research_retrieval_workflow",
            status="completed",
            evidence=[
                Evidence(
                    evidence_id="report-evidence",
                    task_id="research:report",
                    source_type="iwencai.report.search",
                    provider="iwencai",
                ),
                Evidence(
                    evidence_id="report-evidence-2",
                    task_id="research:report",
                    source_type="iwencai.report.search",
                    provider="iwencai",
                ),
            ],
        ),
    ]
    validation = await validate_question_evidence(
        {
            "research_plan": plan,
            "source_results": source_results,
        }
    )
    finalized = await finalize_research(
        {
            "research_plan": plan,
            "source_results": source_results,
            "deep_result": AgentResult(
                task_id="deep-research",
                agent_id="deep_research_agent",
                status="completed",
                answer="研究结论",
            ),
            **validation,
        }
    )
    result = finalized["result"]

    assert result.status == "partial"
    assert result.error_code == "research_question_evidence_partial"
    assert (
        "question:public_disclosures: evidence_source_groups_below_minimum:0<1"
        in result.gaps
    )


async def test_critical_review_counts_independent_source_groups() -> None:
    from agents.research_workflow.workflow import validate_question_evidence

    plan = build_research_plan(
        "研究宁德时代风险",
        {"data_sources": ["local_documents"]},
    )
    chunks = [
        Evidence(
            evidence_id=f"chunk-{index}",
            task_id="deep-research",
            source_type="knowledge.pdf.search",
            provider="local_knowledge",
            metadata={
                "research_question_id": "local_document_facts",
                "source_group": "PDF-AR-CATL-2025",
            },
        )
        for index in range(2)
    ]
    validation = await validate_question_evidence(
        {
            "research_plan": plan,
            "deep_result": AgentResult(
                task_id="deep-research",
                agent_id="deep_research_agent",
                status="completed",
                evidence=chunks,
            ),
        }
    )
    assessments = {
        item.question_id: item for item in validation["question_evidence_assessments"]
    }

    assert assessments["local_document_facts"].evidence_count == 1
    assert assessments["critical_review"].evidence_count == 1
    assert assessments["critical_review"].required_count == 2
    assert assessments["critical_review"].passed is False


async def test_finalize_research_returns_uncovered_when_all_questions_fail() -> None:
    from agents.research_workflow.workflow import (
        finalize_research,
        validate_question_evidence,
    )

    plan = build_research_plan(
        "研究测试公司",
        {"data_sources": ["market"]},
    )
    source_results = [
        AgentResult(
            task_id="research:stock_screening",
            agent_id="stock_screening_agent",
            status="completed",
        )
    ]
    validation = await validate_question_evidence(
        {
            "research_plan": plan,
            "source_results": source_results,
        }
    )
    finalized = await finalize_research(
        {
            "research_plan": plan,
            "source_results": source_results,
            "deep_result": AgentResult(
                task_id="deep-research",
                agent_id="deep_research_agent",
                status="completed",
                answer="证据不足",
            ),
            **validation,
        }
    )

    assert finalized["result"].status == "uncovered"
    assert (
        finalized["result"].error_code
        == "research_question_evidence_uncovered"
    )
