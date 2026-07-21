"""上下文压缩器：按 token 预算滑动窗口 + LLM 增量摘要。

会话级摘要写入独立字段 conversation_summary（thread 内长期保留），
并通过 RemoveMessage 删除已被摘要覆盖的旧消息。
切勿与本轮金融候选答案字段 summary 混用。

第一阶段配置（16K 输入预算）：
- 12K 触发压缩
- 压缩后保留约 8K（含摘要槽位）
- 摘要上限 1.2K
- 按消息 token 从新到旧保留，不再固定轮数
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig

from agents.context_compressor.prompts import SUMMARY_PROMPT, SUMMARY_SHRINK_PROMPT
from agents.context_compressor.tokens import (
    COMPRESS_TRIGGER_TOKENS,
    CONTEXT_TOKEN_BUDGET,
    MAX_SINGLE_MESSAGE_TOKENS,
    POST_COMPRESS_TOKENS,
    SUMMARY_TOKEN_LIMIT,
    capped_message_text,
    estimate_message_tokens,
    estimate_tokens,
    message_text,
    truncate_to_token_limit,
)
from agents.llm import get_router_llm
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="context_compressor")


def _role_label(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "用户"
    if isinstance(message, AIMessage):
        return "助手"
    if isinstance(message, SystemMessage):
        return "系统"
    return "消息"


def _estimate_context_tokens(summary: str, messages: list[AnyMessage]) -> int:
    return estimate_tokens(summary) + sum(estimate_message_tokens(m) for m in messages)


def _last_human_index(messages: list[AnyMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return -1


def select_keep_indices(
    messages: list[AnyMessage],
    *,
    message_token_budget: int,
) -> list[int]:
    """按 token 从最新向前选择保留消息下标；当前用户问题及其后消息始终保留。"""
    if not messages:
        return []

    last_human = _last_human_index(messages)
    if last_human < 0:
        # 无用户消息时仍从尾部按预算保留
        last_human = len(messages)

    keep: set[int] = set(range(last_human, len(messages)))
    used = sum(estimate_message_tokens(messages[i]) for i in keep)

    for index in range(last_human - 1, -1, -1):
        cost = estimate_message_tokens(messages[index])
        if keep and used + cost > message_token_budget:
            break
        keep.add(index)
        used += cost

    return sorted(keep)


def _split_by_keep(
    messages: list[AnyMessage],
    keep_indices: list[int],
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    keep_set = set(keep_indices)
    to_summarize = [m for i, m in enumerate(messages) if i not in keep_set]
    to_keep = [m for i, m in enumerate(messages) if i in keep_set]
    return to_summarize, to_keep


def _truncate_oversized_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """超长单条消息截断后按原 id 回写，避免 SQL/JSON 占满窗口。"""
    updates: list[AnyMessage] = []
    for message in messages:
        text = message_text(message)
        if estimate_tokens(text) <= MAX_SINGLE_MESSAGE_TOKENS:
            continue
        mid = getattr(message, "id", None)
        if not mid:
            continue
        clipped = truncate_to_token_limit(text, MAX_SINGLE_MESSAGE_TOKENS)
        msg_type = type(message)
        try:
            updates.append(msg_type(content=clipped, id=mid))
        except Exception:
            logger.warning("skip truncate for message type={}", msg_type.__name__)
    return updates


async def _enforce_summary_limit(
    summary: str,
    config: RunnableConfig | None = None,
) -> str:
    """摘要超过上限时先尝试 LLM 再压缩，失败则硬截断。"""
    if estimate_tokens(summary) <= SUMMARY_TOKEN_LIMIT:
        return summary

    try:
        result = await get_router_llm().ainvoke(
            [
                (
                    "human",
                    SUMMARY_SHRINK_PROMPT.format(
                        summary_limit=SUMMARY_TOKEN_LIMIT,
                        summary=summary,
                    ),
                )
            ],
            config=config,
        )
        shrunk = (
            result.content
            if isinstance(result.content, str)
            else str(result.content)
        ).strip()
        if shrunk:
            summary = shrunk
    except Exception:
        logger.exception("summary shrink failed, hard truncate")

    if estimate_tokens(summary) > SUMMARY_TOKEN_LIMIT:
        summary = truncate_to_token_limit(summary, SUMMARY_TOKEN_LIMIT)
    return summary


async def _summarize_history(
    existing_summary: str,
    messages: list[AnyMessage],
    config: RunnableConfig | None = None,
) -> str | None:
    """在已有摘要上增量合并本次待压缩消息。

    成功返回摘要文本；失败返回 None（调用方不得删除消息）。
    """
    conversation = "\n".join(
        f"{_role_label(message)}: {capped_message_text(message)}"
        for message in messages
    )
    try:
        result = await get_router_llm().ainvoke(
            [
                (
                    "human",
                    SUMMARY_PROMPT.format(
                        summary_limit=SUMMARY_TOKEN_LIMIT,
                        existing_summary=existing_summary or "无",
                        conversation=conversation,
                    ),
                )
            ],
            config=config,
        )
        summary = (
            result.content
            if isinstance(result.content, str)
            else str(result.content)
        ).strip()
        if not summary:
            logger.warning("summary empty, treat as failure")
            return None
        return await _enforce_summary_limit(summary, config)
    except Exception:
        logger.exception("summary failed")
        return None


async def compress_context(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """按 token 预算压缩上下文。

    - 总上下文 < COMPRESS_TRIGGER_TOKENS：不生成 / 不改写 conversation_summary
    - 超过触发线：增量更新 conversation_summary，按 POST_COMPRESS_TOKENS 倒序保留消息
    - 当前用户问题始终保留；摘要失败则不删除任何消息
    """
    history = list(state.get("messages") or [])
    existing_summary = str(state.get("conversation_summary") or "")

    total_tokens = _estimate_context_tokens(existing_summary, history)
    if total_tokens < COMPRESS_TRIGGER_TOKENS:
        logger.info(
            "compress skipped, tokens={} < trigger={} (budget={})",
            total_tokens,
            COMPRESS_TRIGGER_TOKENS,
            CONTEXT_TOKEN_BUDGET,
        )
        # 仍截断已存在的超长单条，避免工具结果撑爆后续调用
        oversized = _truncate_oversized_messages(history)
        return {"messages": oversized} if oversized else {}

    # 压缩后为目标窗口：摘要槽位 + 近期消息 ≈ POST_COMPRESS_TOKENS
    message_budget = max(1_000, POST_COMPRESS_TOKENS - SUMMARY_TOKEN_LIMIT)
    keep_indices = select_keep_indices(history, message_token_budget=message_budget)
    to_summarize, to_keep = _split_by_keep(history, keep_indices)

    if not to_summarize:
        logger.info("compress skipped, all messages fit post-compress budget")
        oversized = _truncate_oversized_messages(history)
        return {"messages": oversized} if oversized else {}

    summary = await _summarize_history(existing_summary, to_summarize, config)
    if summary is None:
        logger.warning(
            "compress aborted: summary failed, keep all {} messages (tokens={})",
            len(history),
            total_tokens,
        )
        return {}

    removable_messages = [
        message for message in to_summarize if getattr(message, "id", None)
    ]
    if not removable_messages:
        logger.warning("compress aborted: no removable message ids")
        return {}

    if len(removable_messages) < len(to_summarize):
        logger.warning(
            "compress: {} / {} messages missing id, skipped remove",
            len(to_summarize) - len(removable_messages),
            len(to_summarize),
        )

    until_id = removable_messages[-1].id
    kept_tokens = sum(estimate_message_tokens(m) for m in to_keep)
    logger.info(
        "compress: drop={} keep={} summary_tokens≈{} kept_msg_tokens≈{} until={}",
        len(to_summarize),
        len(to_keep),
        estimate_tokens(summary),
        kept_tokens,
        until_id,
    )

    updates: list[AnyMessage] = [
        RemoveMessage(id=message.id) for message in removable_messages
    ]
    updates.extend(_truncate_oversized_messages(to_keep))

    return {
        "conversation_summary": summary,
        "conversation_summary_until": until_id,
        "messages": updates,
    }


__all__ = [
    "COMPRESS_TRIGGER_TOKENS",
    "CONTEXT_TOKEN_BUDGET",
    "POST_COMPRESS_TOKENS",
    "SUMMARY_TOKEN_LIMIT",
    "compress_context",
    "select_keep_indices",
]
