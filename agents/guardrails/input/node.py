"""输入护栏节点：组合确定性检查并生成统一决策。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailStage,
    allow_input,
)
from agents.guardrails.input.injection import check_injection
from agents.guardrails.input.pii import check_pii
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="guardrails")

_HARMFUL_KEYWORDS = ("自杀", "自残", "杀人", "爆炸", "枪支", "毒品", "色情")


def _latest_user_query(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _check_harmful(query: str) -> GuardrailDecision:
    """保留现有敏感内容阻断行为，后续可替换为独立内容策略。"""
    for keyword in _HARMFUL_KEYWORDS:
        if keyword in query:
            return GuardrailDecision(
                action=GuardrailAction.BLOCK,
                stage=GuardrailStage.INPUT,
                reason_code="harmful_content_detected",
                reason=f"检测到敏感词: {keyword}",
                matched_rules=[f"harmful.{keyword}"],
            )
    return allow_input()


def _state_update(decision: GuardrailDecision) -> dict:
    """同时写入新决策合同与旧布尔字段，保证主图平滑迁移。"""
    return {
        "guardrail_decision": decision.model_dump(mode="json"),
        "guardrails_pass": decision.passed,
        "guardrails_reason": decision.reason,
    }


async def guardrails_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """执行输入阶段的确定性护栏检查。"""
    query = _latest_user_query(list(state.get("messages") or []))
    if not query:
        return _state_update(allow_input())

    for check_fn in (check_injection, check_pii, _check_harmful):
        decision = check_fn(query)
        if not decision.passed:
            logger.warning(
                "guardrails blocked: reason_code={} reason={}",
                decision.reason_code,
                decision.reason,
            )
            return _state_update(decision)

    logger.info("guardrails passed")
    return _state_update(allow_input())


def guardrails_edge(state: FinAgentState) -> str:
    """条件边：通过 → memory_recall，拦截 → final_answer。"""
    decision_payload = state.get("guardrail_decision")
    if decision_payload:
        decision = GuardrailDecision.model_validate(decision_payload)
        return "memory_recall" if decision.passed else "final_answer"
    if state.get("guardrails_pass", True):
        return "memory_recall"
    return "final_answer"


__all__ = ["guardrails_edge", "guardrails_node"]
