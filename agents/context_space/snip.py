"""按完整对话或工具调用块裁剪最大历史内容。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from agents.context_compressor.tokens import estimate_tokens, message_text
from agents.context_space.models import SnipMetadata


@dataclass(frozen=True, slots=True)
class _AtomicBlock:
    indices: tuple[int, ...]
    block_type: Literal["dialogue", "tool_call", "message"]
    protected: bool = False


def _last_human_index(messages: list[AnyMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return -1


def _historical_dialogue_blocks(
    messages: list[AnyMessage],
    last_human: int,
) -> list[_AtomicBlock]:
    blocks: list[_AtomicBlock] = []
    index = 0
    while index < max(0, last_human):
        if isinstance(messages[index], SystemMessage):
            blocks.append(_AtomicBlock((index,), "message", protected=True))
            index += 1
            continue
        start = index
        index += 1
        while index < last_human and not isinstance(messages[index], HumanMessage):
            index += 1
        blocks.append(_AtomicBlock(tuple(range(start, index)), "dialogue"))
    return blocks


def _current_turn_blocks(
    messages: list[AnyMessage],
    last_human: int,
) -> list[_AtomicBlock]:
    if last_human < 0:
        return []
    blocks = [_AtomicBlock((last_human,), "message", protected=True)]
    index = last_human + 1
    tool_blocks: list[_AtomicBlock] = []
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            indices = [index]
            index += 1
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                indices.append(index)
                index += 1
            tool_blocks.append(_AtomicBlock(tuple(indices), "tool_call"))
            continue
        blocks.append(_AtomicBlock((index,), "message", protected=index == len(messages) - 1))
        index += 1
    if tool_blocks:
        last = len(tool_blocks) - 1
        blocks.extend(
            _AtomicBlock(block.indices, block.block_type, protected=position == last)
            for position, block in enumerate(tool_blocks)
        )
    return blocks


def _atomic_blocks(messages: list[AnyMessage]) -> list[_AtomicBlock]:
    last_human = _last_human_index(messages)
    if last_human < 0:
        return [
            _AtomicBlock(
                (index,),
                "message",
                protected=isinstance(message, SystemMessage) or index == len(messages) - 1,
            )
            for index, message in enumerate(messages)
        ]
    return [
        *_historical_dialogue_blocks(messages, last_human),
        *_current_turn_blocks(messages, last_human),
    ]


def _block_text(messages: list[AnyMessage], block: _AtomicBlock) -> str:
    return "\n".join(message_text(messages[index]) for index in block.indices)


def _placeholder(message: AnyMessage, *, tokens: int, digest: str, reason: str) -> str:
    if isinstance(message, ToolMessage):
        return (
            "[工具结果已裁剪] "
            f"original_tokens={tokens} content_hash=sha256:{digest} reason={reason}"
        )
    return (
        "[历史内容已裁剪]\n"
        f"类型：{message.type}\n"
        f"原始估算：{tokens} tokens\n"
        f"内容哈希：sha256:{digest}\n"
        f"原因：{reason}"
    )


def snip_largest_removable_block(
    messages: list[AnyMessage],
    *,
    reason: str = "context_budget_exceeded",
) -> tuple[list[AnyMessage], SnipMetadata]:
    """替换最大可裁剪块正文，同时保留消息类型和工具协议字段。"""
    candidates = [block for block in _atomic_blocks(messages) if not block.protected]
    if not candidates:
        return list(messages), SnipMetadata()

    ranked = sorted(
        candidates,
        key=lambda block: (-estimate_tokens(_block_text(messages, block)), block.indices[0]),
    )
    selected = ranked[0]
    raw = _block_text(messages, selected)
    tokens = estimate_tokens(raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    updated = list(messages)
    for index in selected.indices:
        message = messages[index]
        updated[index] = message.model_copy(
            update={"content": _placeholder(message, tokens=tokens, digest=digest, reason=reason)}
        )

    return updated, SnipMetadata(
        changed=True,
        block_type=selected.block_type,
        original_tokens=tokens,
        content_hash=f"sha256:{digest}",
        message_ids=[str(messages[index].id) for index in selected.indices if messages[index].id],
    )


__all__ = ["snip_largest_removable_block"]
