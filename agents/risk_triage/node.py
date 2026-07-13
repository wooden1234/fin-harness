"""Risk Triage 节点：风险分级 + 处置（独立于分类）。"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

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
    history = list(state.get("messages") or [])
    if not history:
        return {"risk_level": "L1", "risk_needs_human": False}

    llm = get_router_llm()
    messages = [
        ("system", RISK_TRIAGE_PROMPT),
        *[("user" if isinstance(m, HumanMessage) else "assistant",
           m.content if isinstance(m.content, str) else str(m.content))
          for m in history],
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
        response = (
            "您的问题已升级为紧急处理，建议立即联系我们的人工客服团队。"
            "客服热线：XXX-XXXX-XXXX（24 小时）。"
        )
        return {
            "risk_level": assessment.risk_level,
            "risk_reason": assessment.reason,
            "risk_needs_human": True,
            "messages": [AIMessage(content=response)],
        }

    return {
        "risk_level": assessment.risk_level,
        "risk_reason": assessment.reason,
        "risk_needs_human": False,
    }


def risk_triage_edge(state: FinAgentState) -> str:
    """条件边：L4 → END（已回复安抚话术），其他 → 继续"""
    if state.get("risk_needs_human", False):
        return "__end__"
    return "plan_agent"
