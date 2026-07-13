"""financial_query_agent 子图公共工具。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agents.states import FinAgentState


def latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    raise ValueError("无用户消息")


def query_from_state(state: FinAgentState) -> str:
    sub_question = state.get("sub_question", "")
    if sub_question:
        return str(sub_question)
    return latest_user_query(list(state.get("messages") or []))


def sub_task_id_from_state(state: FinAgentState) -> str:
    return str(state.get("sub_task_id", ""))


def financial_query_output(
    state: FinAgentState,
    *,
    answer: str,
    context: str | None = None,
    step: str,
    coverage: str = "covered",
    fallback_reason: str = "",
) -> dict:
    """结构化 SQL 查询输出：只返回答案，不写入 citations。

    ``coverage`` 标记证据状态：covered=查到可靠数据 / clarify=需澄清 /
    uncovered=查无或失败（沿证据链降级，禁止改走 FAQ 编数）。
    """
    try:
        query = str(state.get("financial_query_text") or query_from_state(state))
    except ValueError:
        query = str(state.get("sub_question") or "")
    task_result = {
        "sub_task_id": sub_task_id_from_state(state),
        "question": query,
        "type": "financial_query",
        "context": context or answer,
        "coverage": coverage,
    }
    if fallback_reason:
        task_result["fallback_reason"] = fallback_reason
    if coverage == "uncovered":
        task_result["fallback_to_web"] = True
    return {
        "messages": [AIMessage(content=answer)],
        "task_results": [task_result],
        "steps": [step],
    }


def database_failure_output(state: FinAgentState, *, step: str) -> dict:
    return financial_query_output(
        state,
        answer="暂未在结构化财务数据库中找到相关指标。",
        context="（数据库查询失败）",
        step=step,
        coverage="uncovered",
        fallback_reason="financial_query_db_failure",
    )


__all__ = [
    "database_failure_output",
    "financial_query_output",
    "latest_user_query",
    "query_from_state",
    "sub_task_id_from_state",
]
