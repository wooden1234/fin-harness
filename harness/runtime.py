"""产品 AgentManager 单例。"""

from __future__ import annotations

from harness.agent.manager import AgentManager
from harness.llm.deepseek import DeepSeekAdapter
from harness.session.postgres import PostgresSessionStore
from harness.tools.runtime import ToolRuntime

_MANAGER: AgentManager | None = None


def product_manager() -> AgentManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AgentManager(
            store=PostgresSessionStore(),
            llm=DeepSeekAdapter(),
            runtime=ToolRuntime.product(),
        )
    return _MANAGER


def reset_product_manager() -> None:
    global _MANAGER
    _MANAGER = None
