"""Agent 图入口的长期偏好召回节点。"""

from __future__ import annotations

from langgraph.runtime import Runtime

from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState
from app.core.logger import get_logger
from app.services.memory.memory_recall import recall_preferences

logger = get_logger(service="memory_recall")


async def memory_recall_node(
    state: FinAgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict:
    context = runtime.context
    if context is None:
        return {"memory_context": {}}
    try:
        preferences = await recall_preferences(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
        )
    except Exception:
        logger.exception("memory recall failed; continue without long-term memory")
        preferences = {}
    return {"memory_context": preferences}
