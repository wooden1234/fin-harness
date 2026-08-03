"""Agent 侧公共工具循环：bind_tools → 执行 → ToolMessage → 最终回答。

各 node 只声明 tool_ids / system_prompt / llm，不重复写循环。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from agents.context import conversation_messages
from agents.context_compressor.tokens import estimate_tokens, truncate_to_token_limit
from agents.context_space import (
    ContextBudgetPolicy,
    ContextSpaceType,
    ToolLoopContextGovernor,
)
from agents.context_space.events import record_context_event
from agents.runtime_context import (
    AgentRuntimeContext,
    RunHardDeadlineExceeded,
    RunSoftDeadlineExceeded,
)
from app.core.config import settings
from app.core.logger import get_logger
from harness.context import RunContext, build_run_context
from tools import execute_tool, list_bindable_tools, load_all_tools
from tools.core.registry import get_tool_id_by_name, validate_tool_ids

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
        content = json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        content = str(data)
    if estimate_tokens(content) <= 2_000:
        return content
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    clipped = truncate_to_token_limit(content, 1_900)
    return (
        f"{clipped}\n[工具结果已做有界投影] "
        f"original_tokens={estimate_tokens(content)} content_hash=sha256:{digest}"
    )


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
            context = runtime.context if runtime is not None else None
            configured_timeout = float(settings.AGENT_V2_TOOL_SKILL_TIMEOUT_SEC)
            if context is not None and context.unit_timeouts:
                timeout_seconds, timeout_limit = context.execution_timeout_for(
                    "tool_skill",
                    default_seconds=configured_timeout,
                )
            else:
                timeout_seconds, timeout_limit = 0.0, "disabled"
            if timeout_limit == "disabled":
                result = await execute_tool(
                    tool_id,
                    run_context,
                    arguments=args,
                    allowed_tool_ids=allowed_tool_ids,
                )
            elif timeout_seconds <= 0:
                if timeout_limit == "hard":
                    raise RunHardDeadlineExceeded("run_hard_deadline_exceeded")
                if timeout_limit == "soft":
                    raise RunSoftDeadlineExceeded("run_soft_deadline_exceeded")
                raise TimeoutError("tool_skill_timeout")
            else:
                try:
                    async with asyncio.timeout(timeout_seconds):
                        result = await execute_tool(
                            tool_id,
                            run_context,
                            arguments=args,
                            allowed_tool_ids=allowed_tool_ids,
                        )
                except TimeoutError as exc:
                    if timeout_limit == "hard":
                        raise RunHardDeadlineExceeded(
                            "run_hard_deadline_exceeded"
                        ) from exc
                    if timeout_limit == "soft":
                        raise RunSoftDeadlineExceeded(
                            "run_soft_deadline_exceeded"
                        ) from exc
                    raise
            payload: dict[str, Any] = (
                {"ok": True, "data": result.data}
                if result.ok
                else {"ok": False, "error": result.error}
            )
        except (RunHardDeadlineExceeded, RunSoftDeadlineExceeded):
            raise
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            payload = {"ok": False, "error": "tool_skill_timeout"}
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
    governor = ToolLoopContextGovernor(
        llm=llm,
        tools=tools,
        policy=ContextBudgetPolicy(
            explicit_max_tokens=int(settings.CONTEXT_TOOL_LOOP_MAX_TOKENS),
            trigger_ratio=float(settings.CONTEXT_TRIGGER_RATIO),
            target_ratio=float(settings.CONTEXT_TARGET_RATIO),
            admission_ratio=float(settings.CONTEXT_ADMISSION_RATIO),
            approximate_safety_multiplier=float(
                settings.CONTEXT_APPROXIMATE_SAFETY_MULTIPLIER
            ),
            max_compaction_rounds=1,
        ),
        config=config,
    )
    logger.info(
        "{} history_messages={} tools={}",
        agent_name,
        len(working) - 1,
        [tool.name for tool in tools],
    )

    async def _emit_context_event(event_type: str, **details: Any) -> None:
        await record_context_event(
            runtime,
            space_type=ContextSpaceType.TOOL_LOOP,
            event_type=event_type,
            measurement=governor.last_measurement,
            counters=governor.counters,
            details=details,
            agent_id=agent_name,
        )

    async def _prepare_working() -> bool:
        nonlocal working
        before = governor.measure(list(working))
        previous = governor.counters.model_copy(deep=True)
        if before.trigger_exceeded:
            await _emit_context_event("context.threshold_exceeded")
        prepared, admitted = await governor.prepare(list(working))
        working = list(prepared)
        if governor.counters.compaction_round_count > previous.compaction_round_count:
            await _emit_context_event("context.compaction_completed")
        if governor.counters.summary_attempt_count - previous.summary_attempt_count > 1:
            await _emit_context_event("context.summary_repair")
        if governor.counters.snip_count > previous.snip_count:
            await _emit_context_event("context.snip_applied")
        if not admitted:
            logger.warning("{} context admission rejected", agent_name)
            await _emit_context_event("context.admission_rejected", admitted=False)
        return admitted

    async def _recover_overflow() -> bool:
        nonlocal working
        await _emit_context_event("context.provider_overflow")
        recovered, retry = governor.provider_overflow_recovery(list(working))
        if not retry:
            await _emit_context_event("context.degraded", error_code="context_overflow")
            return False
        working = list(recovered)
        await _emit_context_event("context.provider_retry")
        return True

    async def _record_model_result(result: Any) -> None:
        usage = getattr(result, "usage_metadata", None) or {}
        actual = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        measurement = governor.last_measurement
        if measurement is not None and isinstance(actual, int) and actual >= 0:
            measurement = measurement.model_copy(
                update={"actual_input_tokens": actual}
            )
        await record_context_event(
            runtime,
            space_type=ContextSpaceType.TOOL_LOOP,
            event_type="context.model_invocation",
            measurement=measurement,
            counters=governor.counters,
            agent_id=agent_name,
            details={
                "estimate_error_ratio": (
                    abs(actual - measurement.estimated_tokens) / max(1, actual)
                    if measurement is not None and isinstance(actual, int)
                    else None
                )
            },
        )

    async def _stream_final_answer() -> str:
        nonlocal working
        if not await _prepare_working():
            return busy_answer
        parts: list[str] = []
        last_chunk: Any = None
        try:
            async for chunk in llm.astream(working, config=config):
                last_chunk = chunk
                if chunk.content:
                    parts.append(_message_text(chunk.content))
        except ContextOverflowError:
            if not await _recover_overflow():
                return busy_answer
            final = await llm.ainvoke(working, config=config)
            await _record_model_result(final)
            return _message_text(getattr(final, "content", "")).strip() or busy_answer
        if last_chunk is not None:
            await _record_model_result(last_chunk)
        return "".join(parts).strip() or busy_answer

    async def _invoke_final_answer() -> str:
        nonlocal working
        if not await _prepare_working():
            return busy_answer
        try:
            final = await llm.ainvoke(working, config=config)
        except ContextOverflowError:
            if not await _recover_overflow():
                return busy_answer
            final = await llm.ainvoke(working, config=config)
        await _record_model_result(final)
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
            if not await _prepare_working():
                return _response(busy_answer)
            try:
                ai_message = await llm_with_tools.ainvoke(working, config=config)
            except ContextOverflowError:
                if not await _recover_overflow():
                    return _response(busy_answer)
                ai_message = await llm_with_tools.ainvoke(working, config=config)
            await _record_model_result(ai_message)
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
    except (
        asyncio.CancelledError,
        RunHardDeadlineExceeded,
        RunSoftDeadlineExceeded,
    ):
        raise
    except Exception:
        logger.exception("{} llm invoke failed", agent_name)
        return _response(busy_answer)

    return _response(answer)


__all__ = [
    "DEFAULT_BUSY_ANSWER",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "run_with_tools",
]
