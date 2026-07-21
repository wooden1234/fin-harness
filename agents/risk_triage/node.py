"""Risk Triage 节点：风险分级 + 处置（独立于分类）。"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.context import conversation_messages
from agents.llm import get_router_llm
from agents.risk_triage.models import RiskAssessment
from agents.risk_triage.prompts import RISK_TRIAGE_PROMPT
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="risk_triage")


async def risk_triage_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """独立的风险评估与处置节点"""
    history = conversation_messages(state)
    if not history:
        return {"risk_level": "L1", "risk_needs_human": False}

    llm = get_router_llm()
    messages = [
        ("system", RISK_TRIAGE_PROMPT),
        *[
            (
                "system"
                if isinstance(m, SystemMessage)
                else ("user" if isinstance(m, HumanMessage) else "assistant"),
                m.content if isinstance(m.content, str) else str(m.content),
            )
            for m in history
        ],
    ]

    try:
        assessment = cast(
            RiskAssessment,
            await llm.with_structured_output(
                RiskAssessment, method="json_mode"
            ).ainvoke(messages, config=config),
        )
    except Exception:
        logger.exception("risk assessment failed, default to L1")
        assessment = RiskAssessment(risk_level="L1", reason="评估失败，默认放行")

    logger.info("risk={} needs_human={}", assessment.risk_level, assessment.needs_human)

    if assessment.needs_human or assessment.risk_level == "L4":
        return {
            "risk_level": assessment.risk_level,
            "risk_reason": assessment.reason,
            "risk_needs_human": True,
        }

    return {
        "risk_level": assessment.risk_level,
        "risk_reason": assessment.reason,
        "risk_needs_human": False,
    }


def risk_triage_edge(state: FinAgentState) -> str:
    """条件边：需人工 → final_answer 统一收口，其他 → 继续。"""
    if state.get("risk_needs_human", False):
        return "final_answer"
    return "plan_agent"
