"""Agent 侧公共工具循环：bind_tools → 执行 → ToolMessage → 最终回答。

各 node 只声明 tool_ids / system_prompt / llm，不重复写循环。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agents.context import conversation_messages
from app.core.logger import get_logger
from harness.context import RunContext, build_run_context
from tools import execute_tool, list_bindable_tools, load_all_tools
from tools.registry import get_tool_id_by_name

logger = get_logger(service="tool_runtime")

DEFAULT_MAX_TOOL_ROUNDS = 3
DEFAULT_BUSY_ANSWER = "哎呀，我这边有点忙不过来了～ 稍等一下再试试？"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _tool_result_content(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        return str(data)


def _run_context_from_config(
    config: RunnableConfig | None,
    *,
    agent_name: str,
) -> RunContext:
    run_context = build_run_context(metadata={"agent": agent_name})
    configurable = (config or {}).get("configurable") or {}
    if isinstance(configurable, dict):
        run_context.user_id = configurable.get("user_id") or run_context.user_id
        run_context.conversation_id = (
            configurable.get("thread_id")
            or configurable.get("conversation_id")
            or run_context.conversation_id
        )
    return run_context


async def _execute_tool_calls(
    ai_message: AIMessage,
    *,
    config: RunnableConfig | None,
    agent_name: str,
) -> list[ToolMessage]:
    """执行一轮 AIMessage.tool_calls，返回对应 ToolMessage 列表。"""
    run_context = _run_context_from_config(config, agent_name=agent_name)
    tool_messages: list[ToolMessage] = []

    for call in ai_message.tool_calls or []:
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        try:
            tool_id = get_tool_id_by_name(name)
            result = await execute_tool(tool_id, run_context, **args)
            payload: dict[str, Any] = (
                {"ok": True, "data": result.data}
                if result.ok
                else {"ok": False, "error": result.error}
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("{} tool failed name={}", agent_name, name)
            payload = {"ok": False, "error": str(exc)}

        tool_messages.append(
            ToolMessage(content=_tool_result_content(payload), tool_call_id=call_id)
        )
        logger.info("{} tool_call name={} ok={}", agent_name, name, payload.get("ok"))

    return tool_messages


async def run_with_tools(
    state: Mapping[str, Any],
    *,
    llm: BaseChatModel,
    system_prompt: str,
    tool_ids: Sequence[str] = (),
    config: RunnableConfig | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    busy_answer: str = DEFAULT_BUSY_ANSWER,
    agent_name: str = "agent",
    stream_final: bool = True,
) -> dict[str, Any]:
    """带工具的一轮对话：按需调用工具，返回 ``{"messages": [AIMessage(...)]}``。

    - ``tool_ids`` 为空时退化为普通 LLM 调用。
    - 工具轮用 ``ainvoke``（不向用户流式推 tool_calls）。
    - 最终回答默认 ``astream``（含「未调用工具、直接作答」）。
    """
    load_all_tools()
    ids = [str(item) for item in tool_ids if str(item).strip()]
    tools = list_bindable_tools(tool_ids=ids) if ids else []
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    working: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *conversation_messages(state),
    ]
    logger.info(
        "{} history_messages={} tools={}",
        agent_name,
        len(working) - 1,
        [tool.name for tool in tools],
    )

    async def _stream_final_answer() -> str:
        parts: list[str] = []
        async for chunk in llm.astream(working, config=config):
            if chunk.content:
                parts.append(_message_text(chunk.content))
        return "".join(parts).strip() or busy_answer

    async def _invoke_final_answer() -> str:
        final = await llm.ainvoke(working, config=config)
        return _message_text(getattr(final, "content", "")).strip() or busy_answer

    try:
        # 未绑定任何工具：直接生成最终回答
        if not tools:
            answer = (
                await _stream_final_answer()
                if stream_final
                else await _invoke_final_answer()
            )
            return {"messages": [AIMessage(content=answer)]}

        # 有工具：先 ainvoke 决策/执行，最终回答再统一流式
        for round_idx in range(max(1, int(max_tool_rounds or 1))):
            ai_message = await llm_with_tools.ainvoke(working, config=config)
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(
                    content=_message_text(getattr(ai_message, "content", ""))
                )

            if not ai_message.tool_calls:
                # 不再在此处直接返回 ainvoke 文本；落到下方统一最终回答
                break

            working.append(ai_message)
            tool_messages = await _execute_tool_calls(
                ai_message,
                config=config,
                agent_name=agent_name,
            )
            working.extend(tool_messages)
            logger.info(
                "{} tool_round={} calls={}",
                agent_name,
                round_idx + 1,
                len(tool_messages),
            )

        answer = (
            await _stream_final_answer()
            if stream_final
            else await _invoke_final_answer()
        )
    except Exception:
        logger.exception("{} llm invoke failed", agent_name)
        return {"messages": [AIMessage(content=busy_answer)]}

    return {"messages": [AIMessage(content=answer)]}


__all__ = [
    "DEFAULT_BUSY_ANSWER",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "run_with_tools",
]
