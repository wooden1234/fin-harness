"""Harness 统一运行入口。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents.checkpoint import make_thread_config
from agents.graph import get_graph
from harness.context import RunContext, build_run_context
from harness.policy import pre_check


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

    graph = get_graph()
    config = (
        make_thread_config(
            conversation_id,
            user_id=run_context.user_id,
        )
        if conversation_id is not None
        else {"configurable": {"thread_id": run_context.trace_id}}
    )
    return await graph.ainvoke({"messages": [HumanMessage(content=query)]}, config)
