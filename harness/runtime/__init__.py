"""Agent execution layer.

This package is the stable boundary for the mechanics of running an Agent.
The legacy modules under :mod:`harness.agent`, :mod:`harness.session`, and
:mod:`harness.tools` remain available for compatibility during migration.
"""

from harness.session.store import InMemorySessionStore, SessionStore
from harness.tools.runtime import ToolRuntime
from harness.runtime.context import request_header

_MANAGER = None


def product_manager():
    """Return the product composition root (kept here for import compatibility)."""
    global _MANAGER
    if _MANAGER is None:
        from harness.llm.deepseek import DeepSeekAdapter
        from harness.session.postgres import PostgresSessionStore
        from harness.control.manager import AgentManager

        _MANAGER = AgentManager(
            store=PostgresSessionStore(),
            llm=DeepSeekAdapter(),
            runtime=ToolRuntime.product(),
        )
    return _MANAGER


def reset_product_manager() -> None:
    global _MANAGER
    _MANAGER = None


def __getattr__(name: str):
    # Agent 留在公开 API，但不能在包初始化时回导入 loop（loop → runtime.context → 本模块）。
    if name == "Agent":
        from harness.agent.loop import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Agent",
    "InMemorySessionStore",
    "SessionStore",
    "ToolRuntime",
    "product_manager",
    "reset_product_manager",
    "request_header",
]
