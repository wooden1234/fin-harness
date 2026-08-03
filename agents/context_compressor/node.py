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

import json
from typing import Any, Mapping

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.context_compressor.prompts import (
    GENERAL_SUMMARY_PROMPT,
    GENERAL_SUMMARY_SHRINK_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_SHRINK_PROMPT,
    STRUCTURED_SUMMARY_PATCH_PROMPT,
    STRUCTURED_SUMMARY_REPAIR_PROMPT,
)
from agents.context_compressor.models import ConversationSummaryPatch, ConversationSummaryV2
from agents.context_compressor.structured import (
    apply_summary_patch,
    parse_summary_v2,
    render_summary_v2,
)
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
from agents.context_space import ContextBudgetPolicy, ContextCounters, ContextSpaceType
from agents.context_space.budget import measure_context
from agents.context_space.events import record_context_event
from agents.context_space.snip import snip_largest_removable_block
from agents.llm import get_router_llm
from agents.runtime_context import AgentRuntimeContext

from app.core.config import settings
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


def _structured_mode() -> str:
    mode = str(settings.CONTEXT_STRUCTURED_SUMMARY_MODE or "off").strip().lower()
    if mode not in {"off", "shadow", "on"}:
        logger.warning("unknown structured summary mode={}, fallback off", mode)
        return "off"
    return mode


def _last_human_index(messages: list[AnyMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return -1


def _historical_atomic_blocks(
    messages: list[AnyMessage],
    *,
    end: int,
) -> list[list[int]]:
    """把旧历史按完整用户轮次分块，系统消息始终排除在压缩范围外。"""
    blocks: list[list[int]] = []
    current: list[int] = []
    for index, message in enumerate(messages[:end]):
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, HumanMessage) and current:
            blocks.append(current)
            current = []
        current.append(index)
    if current:
        blocks.append(current)
    return blocks


def select_keep_indices(
    messages: list[AnyMessage],
    *,
    message_token_budget: int,
) -> list[int]:
    """按 token 从最新向前选择保留消息下标；当前用户问题及其后消息始终保留。"""
    if not messages:
        return []

    last_human = _last_human_index(messages)
    protected_tail = last_human if last_human >= 0 else len(messages)
    keep: set[int] = {
        index
        for index, message in enumerate(messages)
        if isinstance(message, SystemMessage) or index >= protected_tail
    }
    used = sum(estimate_message_tokens(messages[i]) for i in keep)

    for block in reversed(_historical_atomic_blocks(messages, end=protected_tail)):
        cost = sum(estimate_message_tokens(messages[index]) for index in block)
        if keep and used + cost > message_token_budget:
            break
        keep.update(block)
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
        try:
            updates.append(message.model_copy(update={"content": clipped}))
        except Exception:
            logger.warning("skip truncate for message type={}", type(message).__name__)
    return updates


def _summary_prompts_for_lane(execution_lane: str) -> tuple[str, str]:
    """按执行档选择摘要/再压缩 prompt。"""
    if str(execution_lane or "").strip() == "general":
        return GENERAL_SUMMARY_PROMPT, GENERAL_SUMMARY_SHRINK_PROMPT
    return SUMMARY_PROMPT, SUMMARY_SHRINK_PROMPT


async def _enforce_summary_limit(
    summary: str,
    config: RunnableConfig | None = None,
    counters: ContextCounters | None = None,
    *,
    shrink_prompt: str = SUMMARY_SHRINK_PROMPT,
) -> str:
    """摘要超过上限时先尝试 LLM 再压缩，失败则硬截断。"""
    if estimate_tokens(summary) <= SUMMARY_TOKEN_LIMIT:
        return summary

    try:
        if counters is not None:
            counters.summary_attempt_count += 1
        result = await get_router_llm().ainvoke(
            [
                (
                    "human",
                    shrink_prompt.format(
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
    counters: ContextCounters | None = None,
    *,
    summary_prompt: str = SUMMARY_PROMPT,
    shrink_prompt: str = SUMMARY_SHRINK_PROMPT,
) -> str | None:
    """在已有摘要上增量合并本次待压缩消息。

    成功返回摘要文本；失败返回 None（调用方不得删除消息）。
    """
    conversation = "\n".join(
        f"{_role_label(message)}: {capped_message_text(message)}"
        for message in messages
    )
    try:
        if counters is not None:
            counters.summary_attempt_count += 1
        result = await get_router_llm().ainvoke(
            [
                (
                    "human",
                    summary_prompt.format(
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
        return await _enforce_summary_limit(
            summary,
            config,
            counters,
            shrink_prompt=shrink_prompt,
        )
    except Exception:
        logger.exception("summary failed")
        return None


async def _summarize_history_v2(
    existing: ConversationSummaryV2 | None,
    legacy_summary: str,
    messages: list[AnyMessage],
    config: RunnableConfig | None = None,
    counters: ContextCounters | None = None,
) -> ConversationSummaryV2 | None:
    """最多两次结构化调用：首次 Patch，加一次结构/引用修复。"""
    conversation = "\n".join(
        f"{_role_label(message)}: {capped_message_text(message)}"
        for message in messages
    )
    original_prompt = STRUCTURED_SUMMARY_PATCH_PROMPT.format(
        structured_summary=json.dumps(
            existing.model_dump(mode="json") if existing else {},
            ensure_ascii=False,
        ),
        legacy_summary=legacy_summary or "无",
        conversation=conversation,
    )
    prompt = original_prompt
    model = get_router_llm().with_structured_output(ConversationSummaryPatch)
    last_error = ""
    for attempt in range(2):
        if counters is not None:
            counters.summary_attempt_count += 1
        try:
            patch = await model.ainvoke([("human", prompt)], config=config)
            if not isinstance(patch, ConversationSummaryPatch):
                patch = ConversationSummaryPatch.model_validate(patch)
            return apply_summary_patch(existing, patch)
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:300]}"
            if attempt == 0:
                logger.warning("structured summary patch failed, repairing: {}", last_error)
                prompt = STRUCTURED_SUMMARY_REPAIR_PROMPT.format(
                    error=last_error,
                    original_prompt=original_prompt,
                )
            else:
                logger.exception("structured summary repair failed")
    return None


async def compress_context(
    state: Mapping[str, Any],
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    """按 token 预算压缩上下文。

    注意：入参类型必须能看到编排层字段（如 execution_lane）。
    不可标注为 FinAgentState，否则 LangGraph 会收窄通道，
    导致后续条件边读不到 execution_lane 而误入 DeepAgent。
    """
    history = list(state.get("messages") or [])
    existing_summary = str(state.get("conversation_summary") or "")
    existing_v2 = parse_summary_v2(state.get("conversation_summary_v2"))
    mode = _structured_mode()
    projected_summary = (
        render_summary_v2(existing_v2) if existing_v2 is not None and mode != "off"
        else existing_summary
    )

    policy = ContextBudgetPolicy(
        explicit_max_tokens=int(settings.CONTEXT_CONVERSATION_MAX_TOKENS),
        trigger_ratio=float(settings.CONTEXT_TRIGGER_RATIO),
        target_ratio=float(settings.CONTEXT_TARGET_RATIO),
        admission_ratio=float(settings.CONTEXT_ADMISSION_RATIO),
        approximate_safety_multiplier=float(
            settings.CONTEXT_APPROXIMATE_SAFETY_MULTIPLIER
        ),
        max_compaction_rounds=1,
    )
    measurement = measure_context(
        history,
        policy=policy,
        model=None,
        extra_texts=[projected_summary],
    )
    total_tokens = measurement.estimated_tokens
    if not measurement.trigger_exceeded:
        logger.info(
            "compress skipped, tokens={} < trigger={} (budget={})",
            total_tokens,
            policy.trigger_tokens(measurement.effective_limit),
            measurement.effective_limit,
        )
        # 仍截断已存在的超长单条，避免工具结果撑爆后续调用
        oversized = _truncate_oversized_messages(history)
        return {"messages": oversized} if oversized else {}

    await record_context_event(
        runtime,
        space_type=ContextSpaceType.CONVERSATION,
        event_type="context.threshold_exceeded",
        measurement=measurement,
    )
    await record_context_event(
        runtime,
        space_type=ContextSpaceType.CONVERSATION,
        event_type="context.compaction_started",
        measurement=measurement,
        counters=ContextCounters(compaction_round_count=1),
    )
    message_budget = max(
        1_000,
        policy.target_tokens(measurement.effective_limit) - SUMMARY_TOKEN_LIMIT,
    )
    keep_indices = select_keep_indices(history, message_token_budget=message_budget)
    to_summarize, to_keep = _split_by_keep(history, keep_indices)

    if not to_summarize:
        logger.info("compress skipped, all messages fit post-compress budget")
        oversized = _truncate_oversized_messages(history)
        return {"messages": oversized} if oversized else {}

    counters = ContextCounters(compaction_round_count=1)
    summary_prompt, shrink_prompt = _summary_prompts_for_lane(
        str(state.get("execution_lane") or "")
    )
    legacy_summary: str | None = None
    structured_summary: ConversationSummaryV2 | None = None
    if mode in {"off", "shadow"}:
        legacy_summary = await _summarize_history(
            existing_summary,
            to_summarize,
            config,
            counters,
            summary_prompt=summary_prompt,
            shrink_prompt=shrink_prompt,
        )
    if mode in {"shadow", "on"}:
        structured_summary = await _summarize_history_v2(
            existing_v2,
            existing_summary,
            to_summarize,
            config,
            counters if mode == "on" else None,
        )

    compression_succeeded = (
        legacy_summary is not None if mode in {"off", "shadow"}
        else structured_summary is not None
    )
    if not compression_succeeded:
        logger.warning(
            "compress aborted: summary failed, keep all {} messages (tokens={})",
            len(history),
            total_tokens,
        )
        oversized = _truncate_oversized_messages(history)
        await record_context_event(
            runtime,
            space_type=ContextSpaceType.CONVERSATION,
            event_type="context.degraded",
            measurement=measurement,
            counters=counters,
            details={"error_code": "summary_failed"},
        )
        return {"messages": oversized} if oversized else {}

    if any(not getattr(message, "id", None) for message in to_summarize):
        logger.warning("compress aborted: removable range contains message without id")
        return {}
    removable_messages = list(to_summarize)

    summary_projection = legacy_summary or (
        render_summary_v2(structured_summary) if structured_summary else ""
    )
    post_measurement = measure_context(
        [AIMessage(content=summary_projection), *to_keep],
        policy=policy,
        model=None,
    )
    snipped_updates: list[AnyMessage] = []
    if post_measurement.admission_exceeded:
        snipped_keep, snip = snip_largest_removable_block(to_keep)
        if snip.changed:
            counters.snip_count += 1
            snipped_updates = [
                updated
                for original, updated in zip(to_keep, snipped_keep, strict=True)
                if original.content != updated.content
            ]
            to_keep = snipped_keep
            post_measurement = measure_context(
                [AIMessage(content=summary_projection), *to_keep],
                policy=policy,
                model=None,
            )
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.CONVERSATION,
                event_type="context.snip_applied",
                measurement=post_measurement,
                counters=counters,
                details={
                    "content_hash": snip.content_hash,
                    "block_type": snip.block_type,
                },
            )
    admission_rejected = post_measurement.admission_exceeded
    if admission_rejected:
        await record_context_event(
            runtime,
            space_type=ContextSpaceType.CONVERSATION,
            event_type="context.admission_rejected",
            measurement=post_measurement,
            counters=counters,
            details={"admitted": False},
        )

    until_id = removable_messages[-1].id
    kept_tokens = sum(estimate_message_tokens(m) for m in to_keep)
    logger.info(
        "compress: drop={} keep={} summary_tokens≈{} kept_msg_tokens≈{} until={}",
        len(to_summarize),
        len(to_keep),
        estimate_tokens(
            legacy_summary
            or (render_summary_v2(structured_summary) if structured_summary else "")
        ),
        kept_tokens,
        until_id,
    )

    updates: list[AnyMessage] = [
        RemoveMessage(id=message.id) for message in removable_messages
    ]
    updates.extend(snipped_updates)
    updates.extend(_truncate_oversized_messages(to_keep))

    result = {
        "conversation_summary_until": until_id,
        "context_admission_rejected": admission_rejected,
        "messages": updates,
    }
    if admission_rejected:
        result["summary"] = (
            "当前输入在压缩和安全裁剪后仍超过上下文窗口，已停止继续扩展任务。"
        )
    if legacy_summary is not None:
        result["conversation_summary"] = legacy_summary
    if structured_summary is not None:
        result["conversation_summary_v2"] = structured_summary.model_dump(mode="json")
    await record_context_event(
        runtime,
        space_type=ContextSpaceType.CONVERSATION,
        event_type="context.compaction_completed",
        measurement=measurement,
        counters=counters,
        details={
            "message_count_before": len(history),
            "message_count_after": len(to_keep),
        },
    )
    if counters.summary_attempt_count > 1 and mode == "on":
        await record_context_event(
            runtime,
            space_type=ContextSpaceType.CONVERSATION,
            event_type="context.summary_repair",
            measurement=measurement,
            counters=counters,
        )
    return result


__all__ = [
    "COMPRESS_TRIGGER_TOKENS",
    "CONTEXT_TOKEN_BUDGET",
    "POST_COMPRESS_TOKENS",
    "SUMMARY_TOKEN_LIMIT",
    "compress_context",
    "select_keep_indices",
]
