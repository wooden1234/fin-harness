"""Harness 统一运行入口。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import HumanMessage

from agents.checkpoint import make_thread_config
from agents.graph_selector import get_selected_graph, select_graph_version
from agents.runtime_context import AgentRuntimeContext
from harness.context import RunContext, build_run_context
from harness.policy import pre_check
from app.core.config import settings


def _runtime_context_from_run_context(
    context: RunContext,
    *,
    conversation_id: str | int | None,
    deadline_seconds: float | None = None,
) -> AgentRuntimeContext:
    """把 Harness 上下文转换为 LangGraph Runtime 上下文。"""
    return AgentRuntimeContext(
        tenant_id=context.tenant_id or "default",
        user_id=context.user_id or "0",
        conversation_id=(
            str(conversation_id)
            if conversation_id is not None
            else context.conversation_id
        ),
        run_id=context.trace_id,
        permissions=tuple(context.permissions),
        deadline_monotonic=(
            time.monotonic() + max(0.0, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        ),
        max_concurrency=settings.AGENT_V2_MAX_CONCURRENCY,
    )


async def run_agent(
    query: str,
    *,
    context: RunContext | None = None,
    conversation_id: str | int | None = None,
) -> dict[str, Any]:
    """运行现有主图，并把运行治理入口集中到 Harness。"""
    run_context = context or build_run_context(
        conversation_id=str(conversation_id) if conversation_id is not None else None
    )
    pre_check(run_context)

    graph_version = select_graph_version(
        conversation_id=conversation_id or run_context.trace_id,
        user_id=run_context.user_id,
        tenant_id=run_context.tenant_id,
    )
    graph = get_selected_graph(graph_version)
    runtime_context = _runtime_context_from_run_context(
        run_context,
        conversation_id=conversation_id,
        deadline_seconds=(
            settings.AGENT_V2_COMPOUND_HARD_DEADLINE_SEC
            + settings.AGENT_V2_FINALIZATION_GRACE_SEC
            if graph_version == "v2"
            else None
        ),
    )
    config = (
        make_thread_config(
            conversation_id,
            user_id=run_context.user_id,
            tenant_id=run_context.tenant_id,
            graph_version=graph_version,
        )
        if conversation_id is not None
        else {
            "configurable": {
                "thread_id": (
                    run_context.trace_id
                    if graph_version == "v1"
                    else f"{run_context.trace_id}:graph:v2"
                ),
                "graph_version": graph_version,
            }
        }
    )
    try:
        if graph_version == "v2":
            async with asyncio.timeout(
                settings.AGENT_V2_COMPOUND_HARD_DEADLINE_SEC
                + settings.AGENT_V2_FINALIZATION_GRACE_SEC
            ):
                return await graph.ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config,
                    context=runtime_context,
                )
        return await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config,
            context=runtime_context,
        )
    except (asyncio.CancelledError, TimeoutError):
        raise
    except Exception:
        # V2 内部错误必须由图自身的 retry/fallback/clarify 策略收敛。
        raise
