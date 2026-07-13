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
) -> dict:
    """结构化 SQL 查询输出：只返回答案，不写入 citations。"""
    try:
        query = str(state.get("financial_query_text") or query_from_state(state))
    except ValueError:
        query = str(state.get("sub_question") or "")
    return {
        "messages": [AIMessage(content=answer)],
        "task_results": [
            {
                "sub_task_id": sub_task_id_from_state(state),
                "question": query,
                "type": "financial_query",
                "context": context or answer,
            }
        ],
        "steps": [step],
    }


def database_failure_output(state: FinAgentState, *, step: str) -> dict:
    return financial_query_output(
        state,
        answer="暂未在结构化财务数据库中找到相关指标，建议查阅年报 PDF 文档获取更多信息。",
        context="（数据库查询失败）",
        step=step,
    )


__all__ = [
    "database_failure_output",
    "financial_query_output",
    "latest_user_query",
    "query_from_state",
    "sub_task_id_from_state",
]
