"""FinalAnswer 节点：统一格式化最终输出"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="final_answer")

KNOWLEDGE_SOURCE_TYPES = frozenset({"faq", "pdf", "web"})


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


async def final_answer_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """统一格式化最终回答，附加引用来源"""

    if state.get("guardrails_pass") is False:
        reason = state.get("guardrails_reason", "输入超出业务范围")
        answer = f"抱歉，{reason}。我只能回答金融相关的问题，请重新提问。"
        return {"messages": [AIMessage(content=answer)], "citations": Overwrite([])}

    if state.get("risk_needs_human", False):
        answer = "您的问题已转人工处理，请稍候。"
        return {"messages": [AIMessage(content=answer)], "citations": Overwrite([])}

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

    citations = _filter_current_turn_citations(state, list(state.get("citations") or []))

    seen = set()
    deduped: list[dict] = []
    for c in citations:
        key = c.get("url") or (c.get("source", ""), c.get("page"), c.get("sub_task_id", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    logger.info(
        "final_answer route={} len={} citations={} deduped={}",
        route,
        len(answer),
        len(citations),
        len(deduped),
    )

    return {
        "messages": [AIMessage(content=answer)],
        "citations": Overwrite(deduped),
        "summary": "",
    }
