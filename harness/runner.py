"""脚本入口：走 Agent loop，不再包装 LangGraph。"""

from __future__ import annotations

from typing import Any

from harness.agent.manager import AgentManager
from harness.context import RunContext, build_run_context
from harness.policy import pre_check
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime


async def run_agent(
    query: str,
    *,
    context: RunContext | None = None,
    conversation_id: str | int | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    run_context = context or build_run_context(
        conversation_id=str(conversation_id) if conversation_id is not None else None
    )
    pre_check(run_context)
    manager = AgentManager(
        store=InMemorySessionStore(),
        llm=llm,
        runtime=ToolRuntime.product(),
    )
    agent = await manager.get(
        tenant_id=run_context.tenant_id or "default",
        user_id=str(run_context.user_id or "0"),
        conversation_id=conversation_id,
    )
    result = await agent.prompt(query)
    return {
        "session_id": result.session_id,
        "run_id": result.run_id,
        "finish_reason": result.finish_reason,
        "published_answer": result.published_answer,
        "follow_ups": result.follow_ups,
        "waiting_approval": result.waiting_approval,
        "approval_id": result.approval_id,
        "error": result.error,
    }
