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
    memory_context = state.get("memory_context") or {}
    turn_preferences = state.get("turn_preferences") or {}

    if not summary and not memory_context and not turn_preferences:
        return history

    system_messages: list[SystemMessage] = []
    if summary:
        system_messages.append(SystemMessage(content=f"[{summary_prefix}]\n{summary}"))
    if memory_context:
        preferences = "\n".join(
            f"- {key}={value}" for key, value in sorted(memory_context.items())
        )
        system_messages.append(
            SystemMessage(
                content=(
                    "[用户长期偏好]\n"
                    f"{preferences}\n"
                    "仅在当前请求未明确指定时参考长期偏好；当前轮用户要求优先。"
                )
            )
        )
    if turn_preferences:
        preferences = "\n".join(
            f"- {key}={value}" for key, value in sorted(turn_preferences.items())
        )
        system_messages.append(
            SystemMessage(
                content=(
                    "[本轮临时要求]\n"
                    f"{preferences}\n"
                    "这些要求只在当前轮生效，并覆盖冲突的长期偏好。"
                )
            )
        )
    return [*system_messages, *history]
