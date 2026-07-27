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
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from agents.context import conversation_messages
from agents.runtime_context import AgentRuntimeContext
from app.core.logger import get_logger
from harness.context import RunContext, build_run_context
from tools import execute_tool, list_bindable_tools, load_all_tools
from tools.registry import get_tool_id_by_name, validate_tool_ids

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


def _run_context_from_runtime(
    runtime: Runtime[AgentRuntimeContext] | None,
    *,
    agent_name: str,
) -> RunContext:
    context = runtime.context if runtime is not None else None
    return build_run_context(
        user_id=getattr(context, "user_id", None),
        tenant_id=getattr(context, "tenant_id", None),
        conversation_id=getattr(context, "conversation_id", None),
        permissions=tuple(getattr(context, "permissions", ()) or ()),
        metadata={"agent": agent_name, "run_id": getattr(context, "run_id", None)},
    )


async def run_with_tools(
    state: Mapping[str, Any],
    *,
    llm: BaseChatModel,
    system_prompt: str,
    tool_ids: Sequence[str] = (),
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    busy_answer: str = DEFAULT_BUSY_ANSWER,
    agent_name: str = "agent",
    stream_final: bool = True,
    return_tool_results: bool = False,
) -> dict[str, Any]:
    """带工具的一轮对话：按需调用工具，返回 ``{"messages": [AIMessage(...)]}``。

    - ``tool_ids`` 为空时退化为普通 LLM 调用。
    - 工具轮用 ``ainvoke``（不向用户流式推 tool_calls）。
    - 最终回答默认 ``astream``（含「未调用工具、直接作答」）。
    """
    load_all_tools()
    ids = [str(item) for item in tool_ids if str(item).strip()]
    validate_tool_ids(ids)
    tools = list_bindable_tools(tool_ids=ids) if ids else []
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    allowed_tool_ids = frozenset(ids)
    run_context = _run_context_from_runtime(runtime, agent_name=agent_name)
    tool_results: list[dict[str, Any]] = []

    async def _governed_tool_call(request: Any, _handler: Any) -> ToolMessage:
        """让 ToolNode 负责循环，让项目执行器负责权限和审计边界。"""
        call = request.tool_call
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        tool_id = name
        try:
            tool_id = get_tool_id_by_name(name)
            result = await execute_tool(
                tool_id,
                run_context,
                arguments=args,
                allowed_tool_ids=allowed_tool_ids,
            )
            payload: dict[str, Any] = (
                {"ok": True, "data": result.data}
                if result.ok
                else {"ok": False, "error": result.error}
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("{} tool failed name={}", agent_name, name)
            payload = {"ok": False, "error": str(exc)}
        tool_results.append({"tool_id": tool_id, **payload})
        logger.info("{} tool_call name={} ok={}", agent_name, name, payload.get("ok"))
        return ToolMessage(
            content=_tool_result_content(payload),
            name=name,
            tool_call_id=call_id,
        )

    tool_node = (
        ToolNode(
            tools,
            handle_tool_errors=True,
            awrap_tool_call=_governed_tool_call,
        )
        if tools
        else None
    )

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

    def _response(answer: str) -> dict[str, Any]:
        response: dict[str, Any] = {"messages": [AIMessage(content=answer)]}
        if return_tool_results:
            response["tool_results"] = list(tool_results)
        return response

    try:
        # 未绑定任何工具：直接生成最终回答
        if not tools:
            answer = (
                await _stream_final_answer()
                if stream_final
                else await _invoke_final_answer()
            )
            return _response(answer)

        # 有工具：先 ainvoke 决策/执行，最终回答再统一流式
        for round_idx in range(max(1, int(max_tool_rounds or 1))):
            ai_message = await llm_with_tools.ainvoke(working, config=config)
            if not isinstance(ai_message, AIMessage):
                ai_message = AIMessage(
                    content=_message_text(getattr(ai_message, "content", ""))
                )

            if not ai_message.tool_calls:
                answer = _message_text(ai_message.content).strip() or busy_answer
                return _response(answer)

            working.append(ai_message)
            # 当前 run_with_tools 是一个普通节点，不是独立的 StateGraph 节点；
            # 直接调用 ToolNode 的 Runnable 会缺少 LangGraph 注入的 Runtime，
            # 因此使用其异步节点入口显式传入 Runtime。
            tool_result = await tool_node._afunc(
                {"messages": working},
                config or {},
                Runtime(
                    context=runtime.context if runtime is not None else None,
                ),
            )
            tool_messages = list(tool_result.get("messages") or [])
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
        return _response(busy_answer)

    return _response(answer)


__all__ = [
    "DEFAULT_BUSY_ANSWER",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "run_with_tools",
]
