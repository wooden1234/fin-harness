"""输入护栏节点：组合确定性检查并生成统一决策。"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailSeverity,
    GuardrailStage,
    allow_input,
)
from agents.guardrails.input.harmful import check_harmful
from agents.guardrails.input.injection import check_injection
from agents.guardrails.input.normalization import normalize_input
from agents.guardrails.input.pii import check_pii
from agents.guardrails.input.secrets import check_secrets
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="guardrails")

_ACTION_PRIORITY = {
    GuardrailAction.ALLOW: 0,
    GuardrailAction.REDACT: 1,
    GuardrailAction.CLARIFY: 2,
    GuardrailAction.BLOCK: 3,
    GuardrailAction.ESCALATE: 4,
}

_SEVERITY_PRIORITY = {
    GuardrailSeverity.NONE: 0,
    GuardrailSeverity.LOW: 1,
    GuardrailSeverity.MEDIUM: 2,
    GuardrailSeverity.HIGH: 3,
    GuardrailSeverity.CRITICAL: 4,
}


def _latest_user_message(
    messages: list[AnyMessage],
) -> tuple[int, HumanMessage] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return index, message
    return None


def _aggregate_decisions(
    decisions: list[GuardrailDecision],
) -> GuardrailDecision:
    """聚合全部检查结果，并以最高优先级动作作为主决策。"""
    effective = [
        decision
        for decision in decisions
        if decision.action != GuardrailAction.ALLOW or decision.findings
    ]
    if not effective:
        return allow_input()

    dominant = max(
        effective,
        key=lambda item: (
            _ACTION_PRIORITY[item.action],
            _SEVERITY_PRIORITY[item.severity],
        ),
    )
    findings = [finding for decision in effective for finding in decision.findings]
    matched_rules = list(
        dict.fromkeys(
            rule
            for decision in effective
            for rule in decision.matched_rules
        )
    )
    severity = max(
        (decision.severity for decision in effective),
        key=_SEVERITY_PRIORITY.__getitem__,
    )
    safe_content = next(
        (
            decision.safe_content
            for decision in reversed(effective)
            if decision.safe_content is not None
        ),
        None,
    )
    return GuardrailDecision(
        action=dominant.action,
        stage=GuardrailStage.INPUT,
        severity=severity,
        reason_code=dominant.reason_code,
        reason=dominant.reason,
        safe_content=safe_content,
        findings=findings,
        matched_rules=matched_rules,
        policy_version=dominant.policy_version,
        metadata={
            "finding_count": len(findings),
            "reason_codes": list(
                dict.fromkeys(
                    decision.reason_code
                    for decision in effective
                    if decision.reason_code
                )
            )
        },
    )


def _state_update(
    decision: GuardrailDecision,
    messages: list[AnyMessage] | None = None,
    user_message_index: int | None = None,
) -> dict:
    """同时写入新决策合同与旧布尔字段，保证主图平滑迁移。"""
    update: dict[str, object] = {
        "guardrail_decision": decision.model_dump(mode="json"),
        "guardrails_pass": decision.should_continue,
        "guardrails_reason": decision.reason,
    }
    if (
        decision.safe_content is not None
        and messages is not None
        and user_message_index is not None
    ):
        safe_messages = list(messages)
        safe_messages[user_message_index] = safe_messages[
            user_message_index
        ].model_copy(update={"content": decision.safe_content})
        update["messages"] = Overwrite(safe_messages)
    return update


async def guardrails_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """执行输入阶段的确定性护栏检查。"""
    messages = list(state.get("messages") or [])
    latest_user = _latest_user_message(messages)
    if latest_user is None:
        return _state_update(allow_input())
    user_message_index, user_message = latest_user
    content = user_message.content
    query = content if isinstance(content, str) else str(content)
    if not query:
        return _state_update(allow_input())

    normalized = normalize_input(query)
    decision = _aggregate_decisions(
        [
            check_injection(normalized),
            check_secrets(normalized),
            check_pii(normalized),
            check_harmful(normalized),
        ]
    )
    logger.info(
        "guardrails decision: action={} severity={} reason_code={} findings={}",
        decision.action,
        decision.severity,
        decision.reason_code,
        len(decision.findings),
    )
    return _state_update(decision, messages, user_message_index)


def guardrails_edge(state: FinAgentState) -> str:
    """条件边：通过 → memory_action，拦截 → final_answer。"""
    decision_payload = state.get("guardrail_decision")
    if decision_payload:
        decision = GuardrailDecision.model_validate(decision_payload)
        return "memory_action" if decision.should_continue else "final_answer"
    if state.get("guardrails_pass", True):
        return "memory_action"
    return "final_answer"


__all__ = ["guardrails_edge", "guardrails_node"]
