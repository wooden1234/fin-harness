"""每轮对话初始化节点。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.turn_workspace import begin_turn_workspace


async def init_turn_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """重置仅属于本轮执行的临时工作区。"""
    return begin_turn_workspace()
