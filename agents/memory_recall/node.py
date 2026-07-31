"""Agent 图入口的长期偏好召回节点。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState
from app.core.logger import get_logger
from app.services.memory.memory_recall import recall_preferences
from app.services.memory.memory_command import extract_turn_preferences

logger = get_logger(service="memory_recall")


def _latest_query(state: FinAgentState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def memory_recall_node(
    state: FinAgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict:
    context = runtime.context
    query = _latest_query(state)
    turn_preferences = extract_turn_preferences(query)
    if context is None:
        return {
            "memory_context": {},
            "turn_preferences": turn_preferences,
        }
    try:
        preferences = await recall_preferences(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
            query=query,
        )
    except Exception:
        logger.exception("memory recall failed; continue without long-term memory")
        preferences = {}
    return {
        "memory_context": preferences,
        "turn_preferences": turn_preferences,
    }


__all__ = ["memory_recall_node"]
