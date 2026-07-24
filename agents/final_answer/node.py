"""FinalAnswer 节点：统一格式化最终输出"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.guardrails.output.finance_compliance import (
    review_finance_answer as review_answer,
)
from agents.states import FinAgentState
from app.core.logger import get_logger
from compliance.policies import ComplianceDecision

logger = get_logger(service="final_answer")

KNOWLEDGE_SOURCE_TYPES = frozenset({"faq", "pdf", "web"})
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


def _current_sub_task_ids(state: FinAgentState) -> set[str]:
    return {t.id for t in (state.get("sub_tasks") or []) if getattr(t, "id", None)}


def _filter_current_turn_citations(
    state: FinAgentState,
    citations: list[dict],
) -> list[dict]:
    """保留本轮 worker 产生的可展示引用（faq/pdf/web）。"""
    current_ids = _current_sub_task_ids(state)
    filtered: list[dict] = []
    for citation in citations:
        source_type = str(citation.get("source_type") or "")
        if source_type not in KNOWLEDGE_SOURCE_TYPES:
            continue
        if current_ids and citation.get("sub_task_id") not in current_ids:
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

    if state.get("guardrails_pass") is False:
        reason = state.get("guardrails_reason", "输入超出业务范围")
        answer = f"抱歉，{reason}。我只能回答金融相关的问题，请重新提问。"
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
        "compliance_action": compliance_decision.action,
        "compliance_reason_code": compliance_decision.reason_code,
        "compliance_reason": compliance_decision.reason,
    }
