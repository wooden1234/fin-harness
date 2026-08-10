"""终答节点：澄清透出、三态销单与合规跳过。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.final_answer.node import final_answer_node
from agents.query_rewrite import build_pending_clarification
from compliance.policies import ComplianceDecision


@pytest.mark.asyncio
async def test_final_answer_surfaces_rewrite_uncertain_message() -> None:
    pending = build_pending_clarification("营收多少？", asked_question="请问查哪家公司？")
    update = await final_answer_node(
        {
            "rewrite_status": "uncertain",
            "rewrite_clarification_message": "请问要查哪家公司的营收？",
            "rewrite_reason_codes": ["context_reference"],
            "pending_query_clarification": pending,
            "messages": [HumanMessage(content="营收多少？")],
            "citations": [],
        }
    )
    assert update["messages"][0].content == "请问要查哪家公司的营收？"
    assert update["compliance_reason_code"] == "rewrite_uncertain_passthrough"
    assert "pending_query_clarification" not in update


@pytest.mark.asyncio
async def test_final_answer_expired_clarification_has_default_copy() -> None:
    update = await final_answer_node(
        {
            "rewrite_status": "uncertain",
            "rewrite_clarification_message": "",
            "rewrite_reason_codes": ["clarification_expired"],
            "pending_query_clarification": {},
            "messages": [HumanMessage(content="那个")],
            "citations": [],
        }
    )
    assert "澄清已过期" in update["messages"][0].content
    assert update["compliance_reason_code"] == "rewrite_uncertain_passthrough"


@pytest.mark.asyncio
async def test_final_answer_keeps_pending_on_guardrail_early_exit() -> None:
    update = await final_answer_node(
        {
            "guardrail_decision": {
                "should_continue": False,
                "action": "block",
                "stage": "input",
                "severity": "high",
                "reason_code": "pii_detected",
                "reason": "检测到个人信息",
            },
            "pending_query_clarification": build_pending_clarification("营收多少？"),
            "messages": [HumanMessage(content="我的身份证是110101199001011234")],
            "citations": [],
        }
    )
    assert "pending_query_clarification" not in update
    assert "敏感个人信息" in update["messages"][0].content


@pytest.mark.asyncio
async def test_final_answer_keeps_pending_on_context_admission_rejected() -> None:
    update = await final_answer_node(
        {
            "execution_lane": "deep",
            "context_admission_rejected": True,
            "summary": "上下文过长，请精简后再问。",
            "pending_query_clarification": build_pending_clarification("营收多少？"),
            "messages": [HumanMessage(content="茅台")],
            "citations": [],
        }
    )
    assert "pending_query_clarification" not in update


@pytest.mark.asyncio
async def test_final_answer_keeps_pending_on_memory_action_early_exit() -> None:
    update = await final_answer_node(
        {
            "memory_action_handled": True,
            "messages": [AIMessage(content="已记住你的偏好。", id="mem-1")],
            "pending_query_clarification": build_pending_clarification("营收多少？"),
            "citations": [],
        }
    )
    assert update["messages"][0].content == "已记住你的偏好。"
    assert update["compliance_reason_code"] == "memory_action_passthrough"
    assert "pending_query_clarification" not in update


@pytest.mark.asyncio
async def test_final_answer_keeps_pending_on_deep_clarify() -> None:
    update = await final_answer_node(
        {
            "execution_lane": "deep",
            "route": "main",
            "execution_status": "clarify",
            "summary": "请补充公司名称。",
            "pending_query_clarification": build_pending_clarification(
                "营收多少？",
                asked_question="请补充公司名称。",
            ),
            "messages": [HumanMessage(content="营收多少？")],
            "citations": [],
        }
    )
    assert update["messages"][0].content == "请补充公司名称。"
    assert update["compliance_reason_code"] == "deep_clarify_passthrough"
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
            "pending_query_clarification": build_pending_clarification("营收多少？"),
            "messages": [HumanMessage(content="贵州茅台营收多少？")],
            "citations": [],
        }
    )
    assert update["pending_query_clarification"] == {}
    assert update["messages"][0].content.startswith("贵州茅台")


@pytest.mark.asyncio
async def test_quality_gate_writes_pending_on_clarify() -> None:
    from agents.main_deep_agent.contracts import MainAgentResponse
    from agents.main_deep_agent.quality.validator import main_evidence_quality_gate

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
    assert update["execution_status"] == "clarify"
    pending = update["pending_query_clarification"]
    assert pending["original_query"] == "营收多少？"
    assert pending["asked_question"] == "请补充公司名称。"
    assert pending["remaining_turns"] == 2
    assert pending["target_lane"] == "deep"
