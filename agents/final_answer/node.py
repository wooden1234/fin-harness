"""FinalAnswer 节点：统一格式化最终输出"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.guardrails.contracts import GuardrailAction, GuardrailDecision
from agents.guardrails.output.finance_compliance import (
    review_finance_answer as review_answer,
)
from agents.states import FinAgentState
from app.core.logger import get_logger
from compliance.policies import ComplianceDecision

logger = get_logger(service="final_answer")

KNOWLEDGE_SOURCE_TYPES = frozenset(
    {
        "faq",
        "pdf",
        "web",
        "iwencai",
        "financial_db",
        "financial_fact",
        "financial_query",
    }
)
COMPLIANCE_BLOCKED_ANSWER = (
    "抱歉，我不能提供保证收益、确定涨跌或直接交易指令。"
    "我可以继续为您整理相关公开信息、数据依据和风险因素。"
)
COMPLIANCE_REVIEW_ERROR_ANSWER = (
    "抱歉，当前回答未能通过合规审查，请稍后重试或联系人工服务。"
)
COMPLIANCE_ESCALATED_ANSWER = "该问题需要人工进一步审核，请联系人工服务。"
ROUTE_CLARIFICATION_ANSWER = (
    "我暂时无法明确判断您的查询目标。"
    "请补充更具体的对象和问题，例如公司或产品名称、年份、指标或查询口径。"
)
GUARDRAIL_RESPONSES = {
    "prompt_injection_detected": (
        "无法执行修改系统规则、披露内部指令或绕过安全限制的请求。"
        "请直接描述需要解决的金融问题。"
    ),
    "prompt_injection_suspected": (
        "你的请求可能包含角色切换或指令覆盖内容。"
        "请去除这些内容，并直接描述需要解决的金融问题。"
    ),
    "pii_detected": (
        "输入中包含敏感个人信息，请先完成脱敏后再提交。"
    ),
    "pii_redacted": "输入中的敏感个人信息已脱敏。",
    "secret_detected": (
        "输入中包含密码、Token、验证码、私钥或其他凭据，请删除秘密值后重新提交。"
    ),
    "harmful_content_detected": (
        "无法协助处理该高风险内容。你可以改为咨询相关风险、法规或公开信息。"
    ),
    "harmful_instructions_detected": (
        "无法协助提供高风险行为的实施方法。"
        "你可以改为咨询相关风险、预防措施、法规或公开信息。"
    ),
    "harmful_content_needs_context": (
        "你的请求涉及敏感内容，但目的尚不明确。"
        "请说明你需要的是风险分析、法规信息、预防建议还是公开资料。"
    ),
    "self_harm_crisis_detected": (
        "如果你或他人正面临立即危险，请立即联系当地紧急服务，"
        "并尽快联系身边可信任的人或专业支持人员。"
    ),
}


def _guardrail_response(state: FinAgentState) -> str | None:
    """根据结构化原因返回准确且不暴露命中内容的话术。"""
    decision_payload = state.get("guardrail_decision")
    if decision_payload:
        decision = GuardrailDecision.model_validate(decision_payload)
        if decision.should_continue:
            return None
        reason_response = GUARDRAIL_RESPONSES.get(decision.reason_code)
        if reason_response is not None:
            return reason_response
        if decision.action == GuardrailAction.ESCALATE:
            return "该请求需要人工进一步审核，请联系人工服务。"
        return "当前请求未通过安全检查，请调整内容后重新提交。"

    if state.get("guardrails_pass") is False:
        reason = state.get("guardrails_reason", "输入超出业务范围")
        return f"抱歉，{reason}。请调整内容后重新提交。"
    return None


def _current_sub_task_ids(state: FinAgentState) -> set[str]:
    task_ids = {
        t.id for t in (state.get("sub_tasks") or []) if getattr(t, "id", None)
    }
    task_plan = state.get("task_plan")
    if task_plan is not None:
        task_ids.update(
            task.task_id
            for task in task_plan.tasks
            if getattr(task, "task_id", None)
        )
    task_ids.update(
        result.task_id
        for result in (state.get("agent_results") or [])
        if getattr(result, "task_id", None)
    )
    task_ids.update(
        evidence.task_id
        for evidence in (state.get("evidence") or [])
        if getattr(evidence, "task_id", None)
    )
    return task_ids


def _has_citation_provenance(citation: dict) -> bool:
    """确认 Citation 至少包含可展示的来源信息。"""
    source_type = str(citation.get("source_type") or "")
    if not source_type:
        return False
    if source_type in KNOWLEDGE_SOURCE_TYPES or source_type.startswith("iwencai."):
        return True
    return any(
        citation.get(key)
        for key in (
            "source",
            "title",
            "url",
            "doc_id",
            "document_id",
            "table_id",
            "source_cell_id",
        )
    )


def _filter_current_turn_citations(
    state: FinAgentState,
    citations: list[dict],
) -> list[dict]:
    """按统一 Evidence 契约保留本轮可展示引用，并兼容旧结果格式。"""
    current_ids = _current_sub_task_ids(state)
    constrained_answer = state.get("constrained_answer")
    used_evidence_ids: set[str] | None = None
    if constrained_answer is not None:
        statements = (
            constrained_answer.statements
            if hasattr(constrained_answer, "statements")
            else constrained_answer.get("statements", [])
        )
        used_evidence_ids = {
            str(evidence_id)
            for statement in statements
            for evidence_id in (
                statement.evidence_ids
                if hasattr(statement, "evidence_ids")
                else statement.get("evidence_ids", [])
            )
        }
    filtered: list[dict] = []
    for citation in citations:
        if not _has_citation_provenance(citation):
            continue
        if current_ids and citation.get("sub_task_id") not in current_ids:
            continue
        if (
            used_evidence_ids is not None
            and str(citation.get("evidence_id") or "") not in used_evidence_ids
        ):
            continue
        filtered.append(citation)
    return filtered


def _review_final_answer(answer: str) -> tuple[str, ComplianceDecision]:
    """审查最终候选答案；审查服务异常时默认阻断。"""
    try:
        decision = review_answer(answer)
    except Exception:
        logger.exception("final answer compliance review failed")
        decision = ComplianceDecision(
            action="block",
            reason_code="compliance_review_error",
            reason="合规审查异常",
        )
        return COMPLIANCE_REVIEW_ERROR_ANSWER, decision

    if decision.action == "pass":
        return answer, decision
    if decision.action == "rewrite":
        if decision.safe_answer:
            return decision.safe_answer, decision
        logger.error("compliance rewrite missing safe_answer")
        fallback = ComplianceDecision(
            action="block",
            reason_code="compliance_rewrite_empty",
            reason="合规改写结果为空",
        )
        return COMPLIANCE_REVIEW_ERROR_ANSWER, fallback
    if decision.action == "escalate":
        return COMPLIANCE_ESCALATED_ANSWER, decision
    return COMPLIANCE_BLOCKED_ANSWER, decision


async def final_answer_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """统一格式化最终回答，附加引用来源"""

    force_empty_citations = False
    guardrail_response = _guardrail_response(state)

    if guardrail_response is not None:
        answer = guardrail_response
        force_empty_citations = True
    elif state.get("supervisor_action") in {"rewrite", "clarify"}:
        answer = ROUTE_CLARIFICATION_ANSWER
        force_empty_citations = True
    else:
        route = state.get("route", "general")
        answer = ""

        if route == "general":
            for msg in reversed(list(state.get("messages") or [])):
                if isinstance(msg, AIMessage):
                    answer = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    break
        else:
            answer = state.get("summary", "")
            if not answer:
                for msg in reversed(list(state.get("messages") or [])):
                    if isinstance(msg, AIMessage):
                        answer = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        break

    if not answer:
        answer = "抱歉，我暂时无法回答您的问题，请稍后重试。"

    answer, compliance_decision = _review_final_answer(answer)

    citations = _filter_current_turn_citations(state, list(state.get("citations") or []))

    seen = set()
    deduped: list[dict] = []
    for c in citations:
        key = c.get("url") or (c.get("source", ""), c.get("page"), c.get("sub_task_id", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    if force_empty_citations or compliance_decision.action in {"block", "escalate"}:
        deduped = []

    logger.info(
        "final_answer route={} len={} citations={} deduped={} compliance_action={} reason_code={}",
        state.get("route", "general"),
        len(answer),
        len(citations),
        len(deduped),
        compliance_decision.action,
        compliance_decision.reason_code,
    )

    return {
        "messages": [AIMessage(content=answer)],
        "citations": Overwrite(deduped),
        # summary：本轮候选答案，收口后清空。
        # conversation_summary：多轮会话记忆，此处不得清空。
        "summary": "",
        # 本轮派生字段收口清空，避免跨轮残留。
        "rewritten_query": "",
        "rewrite_status": "",
        "turn_preferences": {},
        "compliance_action": compliance_decision.action,
        "compliance_reason_code": compliance_decision.reason_code,
        "compliance_reason": compliance_decision.reason,
    }
