"""Orchestrator Agent 注册与依赖传递测试。"""

from agents.orchestrator.agent_registry import _dependency_payload, get_agent_spec
from agents.orchestrator.contracts import AgentResult, CandidateSet


def test_v2_specialized_handlers_are_registered() -> None:
    market_spec = get_agent_spec("market_acquisition_workflow")
    research_spec = get_agent_spec("research_retrieval_workflow")
    screening_spec = get_agent_spec("stock_screening_agent")
    compute_spec = get_agent_spec("market.compute")
    research_workflow_spec = get_agent_spec("research_workflow")

    assert market_spec.kind == "workflow"
    assert "iwencai.report.search" in research_spec.capabilities
    assert research_spec.kind == "workflow"
    assert screening_spec.capabilities == ("iwencai.screen",)
    assert compute_spec.kind == "deterministic"
    assert research_workflow_spec.kind == "workflow"
    assert research_workflow_spec.capabilities == ("deep.research",)


def test_dependency_payload_preserves_structured_data() -> None:
    candidates = CandidateSet(
        dataset_id="candidate-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[{"code": "300750", "name": "宁德时代"}],
    )
    source = AgentResult(
        task_id="screen",
        agent_id="stock_screening_agent",
        status="completed",
        answer="共筛选出 1 只候选股",
        structured_data=candidates.model_dump(),
    )

    payload = _dependency_payload(
        [
            source,
        ]
    )

    assert payload[0].structured_data["dataset_id"] == "candidate-1"
    assert payload[0] is not source
