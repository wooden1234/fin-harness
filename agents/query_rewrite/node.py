"""query_rewrite 节点：压缩后、路由前，将追问补全为完整问句。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.query_rewrite.prompts import REWRITE_HUMAN_PROMPT, REWRITE_SYSTEM_PROMPT
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="query_rewrite")

_RECENT_MESSAGE_LIMIT = 8


def _latest_user_query(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _format_recent_dialogue(messages: list[AnyMessage]) -> str:
    """格式化近期对话；最后一条用户消息由 prompt 单独给出，此处排除。"""
    if not messages:
        return "无"

    last_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break

    prior = messages[:last_human_index] if last_human_index >= 0 else messages
    prior = prior[-_RECENT_MESSAGE_LIMIT:]
    if not prior:
        return "无"

    lines: list[str] = []
    for message in prior:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "无"


def _needs_rewrite(
    query: str,
    *,
    existing_summary: str,
    recent_dialogue: str,
) -> bool:
    """无上文可依赖时跳过 LLM，直接沿用原问题。"""
    if not query.strip():
        return False
    if existing_summary.strip():
        return True
    return recent_dialogue.strip() not in {"", "无"}


async def query_rewrite_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """将本轮用户问题改写为完整问句，写入 rewritten_query。"""
    # 下轮开场兜底：若上轮 final_answer 未清掉，先丢掉残值再写本轮结果。
    stale = str(state.get("rewritten_query") or "").strip()
    if stale:
        logger.warning(
            "query_rewrite found uncleared rewritten_query, clearing before rewrite: {}",
            stale[:80],
        )

    history = list(state.get("messages") or [])
    query = _latest_user_query(history)
    if not query:
        return {
            "rewritten_query": "",
            "rewrite_status": "passthrough",
            "steps": ["query_rewrite:empty"],
        }

    existing_summary = str(state.get("conversation_summary") or "").strip()
    recent_dialogue = _format_recent_dialogue(history)

    if not _needs_rewrite(
        query,
        existing_summary=existing_summary,
        recent_dialogue=recent_dialogue,
    ):
        logger.info("query_rewrite skip, use original query")
        return {
            "rewritten_query": query,
            "rewrite_status": "passthrough",
            "steps": ["query_rewrite:passthrough"],
        }

    human = REWRITE_HUMAN_PROMPT.format(
        existing_summary=existing_summary or "无",
        recent_dialogue=recent_dialogue,
        query=query,
    )

    try:
        result = await get_router_llm().ainvoke(
            [
                ("system", REWRITE_SYSTEM_PROMPT),
                ("human", human),
            ],
            config=config,
        )
        rewritten = (
            result.content
            if isinstance(result.content, str)
            else str(result.content)
        ).strip()
    except Exception:
        logger.exception("query_rewrite failed, fallback to original")
        return {
            "rewritten_query": query,
            "rewrite_status": "fallback",
            "steps": ["query_rewrite:fallback"],
        }

    if rewritten == "__UNCERTAIN__" or rewritten.startswith("__UNCERTAIN__"):
        logger.info("query_rewrite uncertain, keep original query")
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "steps": ["query_rewrite:uncertain"],
        }

    if not rewritten:
        rewritten = query

    logger.info(
        "query_rewrite original={} rewritten={}",
        query[:80],
        rewritten[:80],
    )
    return {
        "rewritten_query": rewritten,
        "rewrite_status": "success" if rewritten != query else "passthrough",
        "steps": ["query_rewrite"],
    }
