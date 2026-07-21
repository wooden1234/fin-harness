"""会话上下文组装：摘要与 messages 分离，仅在模型调用时临时合并。"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.messages import AnyMessage, SystemMessage


def conversation_messages(
    state: Mapping[str, Any],
    *,
    summary_prefix: str = "此前对话摘要",
) -> list[AnyMessage]:
    """组装模型调用上下文：有摘要则临时前置 SystemMessage，不写入 checkpoint。"""
    history = list(state.get("messages") or [])
    summary = str(state.get("conversation_summary") or "").strip()

    if not summary:
        return history

    return [
        SystemMessage(content=f"[{summary_prefix}]\n{summary}"),
        *history,
    ]
