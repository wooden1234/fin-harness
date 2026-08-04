"""Orchestrator Agent 注册与依赖传递测试。"""

from agents.orchestrator.agent_registry import _dependency_payload, get_agent_spec
from agents.orchestrator.contracts import AgentResult, CandidateSet


def test_registered_handlers() -> None:
    general_spec = get_agent_spec("general_agent")
    finance_spec = get_agent_spec("finance_agent")

    assert general_spec.capabilities == ()
    assert "faq" in finance_spec.capabilities
    assert finance_spec.evidence_policy.required is True


def test_dependency_payload_preserves_structured_data() -> None:
    candidates = CandidateSet(
        dataset_id="candidate-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[{"code": "300750", "name": "宁德时代"}],
    )
    source = AgentResult(
        task_id="finance",
        agent_id="finance_agent",
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
