"""上下文压缩器：滑动窗口 + LLM 摘要"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="context_compressor")

KEEP_RECENT_TURNS = 4
COMPRESS_THRESHOLD_TURNS = 6

SUMMARY_PROMPT = """请用 1-2 句中文摘要以下对话的核心内容，只记录关键事实：

{conversation}

摘要："""


async def _summarize_history(
    messages: list[AnyMessage],
    config: RunnableConfig | None = None,
) -> str:
    conversation = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in messages
    )
    llm = get_router_llm()
    try:
        result = await llm.ainvoke(
            [("human", SUMMARY_PROMPT.format(conversation=conversation))],
            config=config,
        )
        return (
            result.content
            if isinstance(result.content, str)
            else str(result.content)
        )
    except Exception:
        logger.exception("summary failed")
        return f"（此前共 {len(messages)} 条消息的对话记录）"


def _count_turns(messages: list[AnyMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


async def compress_context(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """上下文压缩：最近 K 轮完整保留，更早的压缩为一行摘要。"""
    history = list(state.get("messages") or [])
    turn_count = _count_turns(history)

    if turn_count <= COMPRESS_THRESHOLD_TURNS:
        logger.info("compress skipped, turns={} <= threshold", turn_count)
        return {}

    user_indices = [
        i for i, m in enumerate(history) if isinstance(m, HumanMessage)
    ]
    if len(user_indices) <= KEEP_RECENT_TURNS:
        return {}

    split_at = user_indices[-(KEEP_RECENT_TURNS)]
    old_messages = history[:split_at]
    recent_messages = history[split_at:]

    summary = await _summarize_history(old_messages, config)
    logger.info(
        "compress: %d messages → summary, %d recent kept",
        len(old_messages),
        len(recent_messages),
    )

    return {
        "messages": [
            SystemMessage(content=f"[对话摘要] {summary}"),
            *recent_messages,
        ],
    }
