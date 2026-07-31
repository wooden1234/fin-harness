"""Root Orchestrator V2 图结构和任务波次测试。"""

import pytest
from langchain_core.messages import HumanMessage

from agents.orchestrator.analyzer import heuristic_profile
from agents.orchestrator.graph import (
    build_plan,
    build_orchestrator_graph,
    clarify,
    detect_evidence_conflicts,
    detect_result_gaps,
    dispatch_wave,
    execute_task,
    evaluate_results,
    _effective_results,
    quality_gate,
    replan,
    route_after_analyze_request,
    route_after_query_rewrite,
    synthesize,
)
from agents.orchestrator.error_policy import classify_error
from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    DomainPlanningScope,
    Evidence,
    QualityReport,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.planner import build_plan_from_profile
from agents.orchestrator.task_identity import validate_task_plan


def test_v2_graph_compiles_with_independent_nodes():
    builder = build_orchestrator_graph()
    graph = builder.compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert "init_turn" in graph.nodes
    assert "memory_action" in graph.nodes
    assert "query_rewrite" in graph.nodes
    assert "analyze_request" in graph.nodes
    assert "execute_task" in graph.nodes
    assert "evaluate_results" in graph.nodes
    assert "final_answer" in graph.nodes
    assert ("__start__", "init_turn") in edges
    assert ("init_turn", "guardrails") in edges
    assert ("guardrails", "memory_action") in edges
    assert "memory_recall" not in graph.nodes
    assert ("memory_action", "context_compressor") in edges
    assert ("build_plan", "memory_plan") in edges
    assert ("memory_plan", "load_task_memories") in edges
    assert ("load_task_memories", "prepare_wave") in edges
    assert builder.branches["replan"]["route_after_replan"].ends == {
        "memory_plan": "memory_plan",
        "synthesize": "synthesize",
    }
    assert ("context_compressor", "query_rewrite") in edges
    assert ("query_rewrite", "analyze_request") in edges
    assert ("analyze_request", "build_plan") in edges
    assert ("analyze_request", "clarify") in edges


def test_build_plan_assigns_task_identity_fields():
    profile = heuristic_profile("查询某公司的财务数据")
    import asyncio

    update = asyncio.run(build_plan({"request_profile": profile}))
    task = update["task_plan"].tasks[0]

    assert task.logical_task_id == task.task_id
    assert task.attempt_id
    assert task.attempt_number == 1
    assert task.idempotency_key.startswith("v2:")


def test_finance_plan_carries_root_domain_scope():
    profile = heuristic_profile("查询某公司的财务数据")
    plan = build_plan_from_profile(profile)
    task = plan.tasks[0]
    scope = DomainPlanningScope.model_validate(
        task.input_data["domain_planning_scope"]
    )

    assert scope.parent_task_id == task.task_id
    assert scope.allowed_capabilities == ["faq", "pdf", "financial_query"]
    assert "web_search" not in scope.allowed_capabilities
    assert "market_event" not in scope.allowed_intents
    assert scope.evidence_policy.required is True


def test_v2_plan_validation_rejects_finance_scope_expansion():
    expanded_scope = DomainPlanningScope(
        parent_task_id="finance",
        parent_logical_task_id="finance",
        allowed_capabilities=["financial_query", "web_search"],
        allowed_intents=["structured_metric", "market_event"],
    )
    plan = TaskPlan(
        plan_id="p-scope-expansion",
        query="查询财务",
        tasks=[
            TaskSpec(
                task_id="finance",
                objective="查询财务",
                agent_id="finance_agent",
                input_data={
                    "domain_planning_scope": expanded_scope.model_dump(
                        mode="json"
                    )
                },
            )
        ],
    )

    import pytest

    with pytest.raises(ValueError, match="domain_scope_capability_mismatch"):
        validate_task_plan(plan)


def test_v2_plan_validation_adds_safe_scope_to_finance_replan_task():
    plan = validate_task_plan(
        TaskPlan(
            plan_id="p-replan-scope",
            query="补充财务证据",
            tasks=[
                TaskSpec(
                    task_id="finance-replan",
                    objective="补充财务证据",
                    agent_id="finance_agent",
                )
            ],
        )
    )
    scope = DomainPlanningScope.model_validate(
        plan.tasks[0].input_data["domain_planning_scope"]
    )

    assert scope.allowed_capabilities == ["faq", "pdf", "financial_query"]
    assert "web_search" not in scope.allowed_capabilities
    assert "market_event" not in scope.allowed_intents


def test_replan_revalidates_suggested_task_capabilities():
    plan = TaskPlan(
        plan_id="p1",
        query="查询财务",
        max_replans=1,
        tasks=[
            TaskSpec(
                task_id="research",
                objective="查询财务",
                agent_id="finance_agent",
            )
        ],
    )
    invalid_suggestion = TaskSpec(
        task_id="unsupported",
        objective="执行未注册能力",
        agent_id="finance_agent",
        required_capabilities=["capability.that.does.not.exist"],
    )

    import asyncio

    update = asyncio.run(
        replan(
            {
                "task_plan": plan,
                "replan_count": 0,
                "quality_report": QualityReport(
                    passed=False,
                    suggested_tasks=[invalid_suggestion],
                ),
                "agent_results": [],
            }
        )
    )

    assert update["next_action"] == "synthesize"
    assert update["execution_status"] == "replan_validation_failed"
    assert "task_plan" not in update


def test_v2_query_rewrite_routes_uncertain_followup_to_clarify():
    assert route_after_query_rewrite({"rewrite_status": "rewrite"}) == "analyze_request"
    assert route_after_query_rewrite({"rewrite_status": "passthrough"}) == "analyze_request"
    assert route_after_query_rewrite({"rewrite_status": "uncertain"}) == "clarify"
    assert route_after_query_rewrite({"rewrite_status": ""}) == "clarify"


def test_v2_analyzer_stops_when_profile_remains_ambiguous():
    ambiguous = heuristic_profile("帮我看看")
    complete = heuristic_profile("贵州茅台 2025 年营收是多少")

    assert route_after_analyze_request({"request_profile": ambiguous}) == "clarify"
    assert route_after_analyze_request({"request_profile": complete}) == "build_plan"
    assert route_after_analyze_request({}) == "clarify"


def test_v3_analyzer_uses_semantics_for_ambiguity_not_agent_selection():
    complete_profile = heuristic_profile("贵州茅台 2025 年营收是多少")

    assert (
        route_after_analyze_request({"request_profile": complete_profile})
        == "build_plan"
    )


def test_clarify_asks_naturally_for_missing_context():
    import asyncio

    output = asyncio.run(
        clarify({"rewrite_reason_codes": ["context_reference", "context_missing"]})
    )

    assert "具体指谁" in output["summary"]
    assert "直接回复名称或代码" in output["summary"]
    assert "本轮已停止处理" not in output["summary"]
    assert output["steps"] == ["orchestrator:clarify_stop"]


def test_clarify_uses_rewrite_llm_generated_message():
    import asyncio

    generated = (
        "我需要先确认“它”指的是哪个标的，因为去年的表现要结合具体对象分析。"
        "例如：**贵州茅台去年怎么样？** 你想查的是哪只股票、基金或指数？"
    )

    output = asyncio.run(
        clarify(
            {
                "messages": [HumanMessage(content="它去年怎么样？")],
                "rewrite_reason_codes": ["context_reference", "context_missing"],
                "rewrite_clarification_message": generated,
            }
        )
    )

    assert output["summary"] == generated


def test_clarify_explains_irrelevant_history():
    import asyncio

    output = asyncio.run(
        clarify({"rewrite_reason_codes": ["context_reference", "context_irrelevant"]})
    )

    assert "没找到能和当前问题对应上的金融对象" in output["summary"]
    assert "贵州茅台" in output["summary"]


def test_clarify_names_safe_multiple_candidates():
    import asyncio

    output = asyncio.run(
        clarify(
            {
                "rewrite_reason_codes": ["multiple_entity_candidates"],
                "pending_query_clarification": {
                    "candidate_entities": ["贵州茅台", "五粮液"]
                },
            }
        )
    )

    assert "贵州茅台、五粮液" in output["summary"]
    assert "直接回复名称或代码" in output["summary"]


def test_clarify_explains_analyzer_missing_fields():
    import asyncio

    output = asyncio.run(
        clarify({"request_profile": heuristic_profile("帮我看看")})
    )

    assert "你想看哪个对象" in output["summary"]
    assert "白酒龙头" in output["summary"]
    assert output["pending_query_clarification"]["original_query"] == "帮我看看"


def test_clarify_uses_analyzer_llm_generated_message():
    import asyncio

    generated = (
        "我还需要一个明确的分析对象。比如：**白酒龙头去年表现怎么样？** "
        "您想看哪只股票、基金、指数或行业？"
    )
    profile = heuristic_profile("帮我看看").model_copy(
        update={"clarification_message": generated}
    )

    output = asyncio.run(clarify({"request_profile": profile}))

    assert output["summary"] == generated


def test_error_action_is_recorded_on_task_failure(monkeypatch):
    import asyncio

    async def fail_invoke(task, **kwargs):
        del task, kwargs
        raise TimeoutError("provider timeout")

    monkeypatch.setattr("agents.orchestrator.graph.invoke_agent", fail_invoke)
    update = asyncio.run(
        execute_task(
            {
                "current_task": TaskSpec(
                    task_id="research",
                    objective="查询研报",
                    agent_id="research_retrieval_workflow",
                )
            }
        )
    )

    result = update["agent_results"][0]
    assert result.error_action == "retry"
    assert result.metadata["retryable"] is True


def test_clarify_error_routes_to_clarify():
    import asyncio

    result = AgentResult(
        task_id="market_compute",
        agent_id="market.compute",
        status="clarify",
        error_code="market_query_plan_missing",
        error_action=classify_error("market_query_plan_missing").action,
    )
    output = asyncio.run(evaluate_results({"agent_results": [result]}))

    assert output["next_action"] == "clarify"


def test_fallback_error_is_not_retried():
    import asyncio

    plan = TaskPlan(
        plan_id="p-fallback",
        query="查询研报",
        tasks=[
            TaskSpec(
                task_id="research",
                objective="查询研报",
                agent_id="research_retrieval_workflow",
            )
        ],
    )
    output = asyncio.run(
        replan(
            {
                "task_plan": plan,
                "replan_count": 0,
                "quality_report": QualityReport(
                    passed=False,
                    failed_task_ids=["research"],
                ),
                "agent_results": [
                    AgentResult(
                        task_id="research",
                        agent_id="research_retrieval_workflow",
                        status="failed",
                        error_code="source_tool_failed",
                        error_action="fallback",
                    )
                ],
            }
        )
    )

    assert output["next_action"] == "synthesize"
    assert output["execution_status"] == "partial"


async def test_retry_result_supersedes_original_failure(monkeypatch):
    async def fake_invoke_agent(task, **kwargs):
        del kwargs
        return AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="completed",
            answer="重试成功",
            evidence=[Evidence(evidence_id="retry-evidence", source_type="web")],
        )

    monkeypatch.setattr("agents.orchestrator.graph.invoke_agent", fake_invoke_agent)
    retry_task = TaskSpec(
        task_id="research:retry:1",
        objective="补充执行：查询研究",
        agent_id="finance_agent",
        metadata={"replan_of": "research"},
    )
    update = await execute_task({"current_task": retry_task})
    retry_result = update["agent_results"][0]

    effective = _effective_results(
        {
            "agent_results": [
                AgentResult(
                    task_id="research",
                    agent_id="finance_agent",
                    status="failed",
                    error_code="timeout",
                ),
                retry_result,
            ]
        }
    )

    assert retry_result.metadata["replan_of"] == "research"
    assert [item.task_id for item in effective] == ["research:retry:1"]


def test_compound_stock_query_creates_independent_research_workflow():
    profile = heuristic_profile("筛选新能源股票并分析前三只的风险")
    plan = build_plan_from_profile(profile)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_id == "research"
    assert plan.tasks[0].agent_id == "research_workflow"
    assert plan.tasks[0].depends_on == []
    assert plan.tasks[0].input_data["data_sources"] == [
        "market",
        "research",
        "finance_rag",
        "local_documents",
    ]


def test_market_compute_receives_previous_candidate_set():
    import asyncio

    candidates = CandidateSet(
        dataset_id="previous-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[{"code": "AAA", "pe_ttm": 20}],
    )
    previous = AgentResult(
        task_id="previous-market",
        agent_id="stock_screening_agent",
        status="completed",
        structured_data=candidates.model_dump(),
    )
    profile = heuristic_profile(
        "从刚才候选股票中按市盈率排序取前5只",
        candidate_set_id="previous-1",
    )
    update = asyncio.run(
        build_plan(
            {
                "request_profile": profile,
                "agent_results": [previous],
            }
        )
    )
    sends = dispatch_wave(
        {
            "task_plan": update["task_plan"],
            "prior_agent_results": update["prior_agent_results"],
            "agent_results": [],
        }
    )

    assert sends[0].arg["current_dependency_results"][0].task_id == "previous-market"


def test_quality_gate_detects_result_gap_and_conflicting_claims():
    state = {
        "agent_results": [
            AgentResult(
                task_id="research",
                agent_id="finance_agent",
                status="failed",
                error_code="provider_timeout",
            )
        ],
        "evidence": [
            Evidence(
                evidence_id="e1",
                source_type="financial_db",
                metadata={"claim_key": "600519.revenue.2025", "claim_value": "1500"},
            ),
            Evidence(
                evidence_id="e2",
                source_type="web",
                metadata={"claim_key": "600519.revenue.2025", "claim_value": "1450"},
            ),
        ],
    }

    assert detect_result_gaps(state) == ["research: provider_timeout"]
    assert detect_evidence_conflicts(state)


def test_quality_gate_does_not_require_evidence_for_general_agent():
    import asyncio

    output = asyncio.run(
        quality_gate(
            {
                "task_plan": TaskPlan(
                    plan_id="p-general",
                    query="你好",
                    tasks=[
                        TaskSpec(
                            task_id="general",
                            objective="你好",
                            agent_id="general_agent",
                        )
                    ],
                ),
                "agent_results": [
                    AgentResult(
                        task_id="general",
                        agent_id="general_agent",
                        status="completed",
                        answer="你好！",
                    )
                ],
                "evidence": [],
            }
        )
    )

    assert output["quality_report"].passed is True
    assert output["quality_report"].missing_evidence == []


def test_quality_gate_requires_finance_evidence_and_provenance():
    import asyncio

    output = asyncio.run(
        quality_gate(
            {
                "task_plan": TaskPlan(
                    plan_id="p-finance",
                    query="查询营收",
                    tasks=[
                        TaskSpec(
                            task_id="finance",
                            objective="查询营收",
                            agent_id="finance_agent",
                        )
                    ],
                ),
                "agent_results": [
                    AgentResult(
                        task_id="finance",
                        agent_id="finance_agent",
                        status="completed",
                        answer="营收增长",
                        evidence=[Evidence(evidence_id="e-1", source_type="pdf")],
                    )
                ],
                "evidence": [],
            }
        )
    )

    assert output["quality_report"].passed is False
    assert "finance: evidence_provenance_missing:1" in output["quality_report"].missing_evidence


def test_quality_gate_requires_structured_market_result():
    import asyncio

    output = asyncio.run(
        quality_gate(
            {
                "task_plan": TaskPlan(
                    plan_id="p-market",
                    query="查询行情",
                    tasks=[
                        TaskSpec(
                            task_id="market",
                            objective="查询行情",
                            agent_id="market_acquisition_workflow",
                        )
                    ],
                ),
                "agent_results": [
                    AgentResult(
                        task_id="market",
                        agent_id="market_acquisition_workflow",
                        status="completed",
                        evidence=[
                            Evidence(
                                evidence_id="e-1",
                                source_type="iwencai.market.query",
                                provider="iwencai",
                            )
                        ],
                    )
                ],
                "evidence": [],
            }
        )
    )

    assert output["quality_report"].passed is False
    assert "market: structured_data_missing" in output["quality_report"].missing_evidence


def test_replan_adds_retry_task_with_budget():
    plan = TaskPlan(
        plan_id="p1",
        query="查询财务",
        max_replans=1,
        tasks=[
            TaskSpec(
                task_id="research",
                objective="查询财务",
                agent_id="finance_agent",
            )
        ],
    )
    state = {
        "task_plan": plan,
        "replan_count": 0,
        "quality_report": QualityReport(
            passed=False,
            failed_task_ids=["research"],
        ),
        "agent_results": [
            AgentResult(
                task_id="research",
                agent_id="finance_agent",
                status="failed",
                error_code="timeout",
            )
        ],
    }

    import asyncio

    update = asyncio.run(replan(state))
    assert update["replan_count"] == 1
    assert update["next_action"] == "schedule"
    retry = update["task_plan"].tasks[-1]
    assert retry.task_id == "research:retry:1"
    assert retry.logical_task_id == "research"
    assert retry.attempt_number == 2
    assert retry.attempt_id
    assert retry.idempotency_key.startswith("v2:research:2:")


def test_synthesize_marks_partial_results_and_unresolved_gaps():
    import asyncio

    result = AgentResult(
        task_id="research",
        agent_id="finance_agent",
        status="partial",
        answer="已完成部分财务查询",
        gaps=["缺少最新机构评级"],
    )
    output = asyncio.run(
        synthesize(
            {
                "agent_results": [result],
                "evidence": [],
                "quality_report": QualityReport(
                    passed=False,
                    missing_evidence=["缺少最新机构评级"],
                ),
            }
        )
    )
    assert "部分完成" in output["summary"]
    assert "缺少最新机构评级" in output["summary"]
