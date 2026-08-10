"""P1：target_lane、延迟销单与多轮澄清链路回归。"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agents.final_answer.node import final_answer_node
from agents.main_deep_agent.contracts import MainAgentResponse
from agents.main_deep_agent.quality.validator import main_evidence_quality_gate
from agents.orchestrator.execution_lane import classify_execution_lane_node
from agents.query_rewrite import build_pending_clarification, query_rewrite_node
from compliance.policies import ComplianceDecision


def test_build_pending_clarification_includes_target_lane() -> None:
    pending = build_pending_clarification(
        "营收多少？",
        asked_question="请问查哪家公司？",
        target_lane="deep",
    )
    assert pending["target_lane"] == "deep"
    assert pending["original_query"] == "营收多少？"
    assert build_pending_clarification("x", target_lane="invalid").get("target_lane") is None
    assert "target_lane" not in build_pending_clarification("x", target_lane="invalid")


@pytest.mark.asyncio
async def test_quality_gate_clarify_sets_target_lane_deep() -> None:
    update = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="营收多少？")],
            "rewritten_query": "营收多少？",
            "rewrite_status": "passthrough",
            "main_agent_response": MainAgentResponse(
                mode="clarify",
                clarification="请补充公司名称。",
            ),
            "evidence": [],
        }
    )
    pending = update["pending_query_clarification"]
    assert pending["target_lane"] == "deep"
    assert pending["asked_question"] == "请补充公司名称。"


@pytest.mark.asyncio
async def test_execution_lane_honors_pending_target_lane_on_resume() -> None:
    update = await classify_execution_lane_node(
        {
            "messages": [HumanMessage(content="茅台")],
            "rewritten_query": "贵州茅台营收多少？",
            "rewrite_status": "rewrite",
            "rewrite_reason_codes": ["clarification_reply"],
            "pending_query_clarification": build_pending_clarification(
                "贵州茅台营收多少？",
                target_lane="deep",
            ),
        }
    )
    assert update["execution_lane"] == "deep"
    assert update["route"] == "deep"


@pytest.mark.asyncio
async def test_query_rewrite_pending_reply_defers_clear(monkeypatch) -> None:
    pending = build_pending_clarification(
        "营收多少？",
        asked_question="请问查哪家公司？",
        target_lane="deep",
        candidate_entities=("茅台",),
    )

    async def fake_rewrite(_prompt: str, _config=None) -> str:
        return "贵州茅台营收多少？"

    monkeypatch.setattr(
        "agents.query_rewrite.node._invoke_rewrite_model",
        fake_rewrite,
    )
    update = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="营收多少？"),
                HumanMessage(content="茅台"),
            ],
            "pending_query_clarification": pending,
        }
    )
    assert update["rewrite_status"] == "rewrite"
    assert "clarification_reply" in update["rewrite_reason_codes"]
    assert update["rewritten_query"] == "贵州茅台营收多少？"
    kept = update["pending_query_clarification"]
    assert kept["original_query"] == "贵州茅台营收多少？"
    assert kept["target_lane"] == "deep"
    assert kept["remaining_turns"] == 2


@pytest.mark.asyncio
async def test_query_rewrite_cancel_clears_pending() -> None:
    update = await query_rewrite_node(
        {
            "messages": [HumanMessage(content="算了")],
            "pending_query_clarification": build_pending_clarification(
                "营收多少？",
                target_lane="deep",
            ),
        }
    )
    assert update["rewrite_reason_codes"] == ["clarification_cancelled"]
    assert update["pending_query_clarification"] == {}


@pytest.mark.asyncio
async def test_query_rewrite_turns_exhausted() -> None:
    pending = build_pending_clarification(
        "营收多少？",
        target_lane="deep",
        remaining_turns=0,
    )
    update = await query_rewrite_node(
        {
            "messages": [HumanMessage(content="那个")],
            "pending_query_clarification": pending,
        }
    )
    assert "clarification_expired" in update["rewrite_reason_codes"]
    assert update["rewrite_status"] == "uncertain"
    assert "澄清已过期" in update["rewrite_clarification_message"]
    assert update["pending_query_clarification"] == {}


@pytest.mark.asyncio
async def test_final_answer_keeps_pending_when_deep_fails() -> None:
    update = await final_answer_node(
        {
            "execution_lane": "deep",
            "route": "main",
            "execution_status": "failed",
            "summary": "暂时无法完成检索。",
            "pending_query_clarification": build_pending_clarification(
                "贵州茅台营收多少？",
                target_lane="deep",
            ),
            "messages": [HumanMessage(content="茅台")],
            "citations": [],
        }
    )
    assert "pending_query_clarification" not in update


@pytest.mark.asyncio
async def test_final_answer_clears_pending_after_deep_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.final_answer.node.review_answer",
        lambda _answer: ComplianceDecision(
            action="pass",
            reason_code="ok",
            reason="pass",
        ),
    )
    update = await final_answer_node(
        {
            "execution_lane": "deep",
            "route": "main",
            "execution_status": "completed",
            "summary": "贵州茅台 2024 年营收约 XXX 亿元。",
            "pending_query_clarification": build_pending_clarification(
                "贵州茅台营收多少？",
                target_lane="deep",
            ),
            "messages": [HumanMessage(content="茅台")],
            "citations": [],
        }
    )
    assert update["pending_query_clarification"] == {}


@pytest.mark.asyncio
async def test_e2e_clarify_resume_success_chain(monkeypatch) -> None:
    """澄清写单 → 补充续办保单并定档 deep → 成功终答销单。"""

    clarify = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="营收多少？")],
            "rewritten_query": "营收多少？",
            "rewrite_status": "passthrough",
            "main_agent_response": MainAgentResponse(
                mode="clarify",
                clarification="请补充公司名称。",
            ),
            "evidence": [],
        }
    )
    pending = clarify["pending_query_clarification"]
    assert pending["target_lane"] == "deep"

    async def fake_rewrite(_prompt: str, _config=None) -> str:
        return "贵州茅台营收多少？"

    monkeypatch.setattr(
        "agents.query_rewrite.node._invoke_rewrite_model",
        fake_rewrite,
    )
    rewrite = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="营收多少？"),
                HumanMessage(content="茅台"),
            ],
            "pending_query_clarification": pending,
        }
    )
    assert "clarification_reply" in rewrite["rewrite_reason_codes"]
    assert rewrite["pending_query_clarification"]["target_lane"] == "deep"

    lane = await classify_execution_lane_node(
        {
            "messages": [HumanMessage(content="茅台")],
            **{k: rewrite[k] for k in (
                "rewritten_query",
                "rewrite_status",
                "rewrite_reason_codes",
                "pending_query_clarification",
            )},
        }
    )
    assert lane["execution_lane"] == "deep"

    monkeypatch.setattr(
        "agents.final_answer.node.review_answer",
        lambda _answer: ComplianceDecision(
            action="pass",
            reason_code="ok",
            reason="pass",
        ),
    )
    final = await final_answer_node(
        {
            "execution_lane": "deep",
            "route": "main",
            "execution_status": "completed",
            "summary": "贵州茅台营收……",
            "pending_query_clarification": rewrite["pending_query_clarification"],
            "messages": [HumanMessage(content="茅台")],
            "citations": [],
        }
    )
    assert final["pending_query_clarification"] == {}
