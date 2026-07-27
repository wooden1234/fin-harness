"""Root Orchestrator V2 图结构和任务波次测试。"""

from agents.orchestrator.analyzer import heuristic_profile
from agents.orchestrator.graph import (
    detect_evidence_conflicts,
    detect_result_gaps,
    quality_gate,
    replan,
    synthesize,
    build_orchestrator_graph,
)
from agents.orchestrator.contracts import (
    AgentResult,
    Evidence,
    QualityReport,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.planner import build_plan_from_profile


def test_v2_graph_compiles_with_independent_nodes():
    graph = build_orchestrator_graph().compile().get_graph()
    assert "analyze_request" in graph.nodes
    assert "execute_task" in graph.nodes
    assert "evaluate_results" in graph.nodes
    assert "final_answer" in graph.nodes


def test_compound_stock_query_creates_dependency_plan():
    profile = heuristic_profile("筛选新能源股票并分析前三只的风险")
    plan = build_plan_from_profile(profile)

    assert [task.task_id for task in plan.tasks] == ["screen", "research"]
    assert plan.tasks[1].depends_on == ["screen"]


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
    assert update["task_plan"].tasks[-1].task_id == "research:retry:1"


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
