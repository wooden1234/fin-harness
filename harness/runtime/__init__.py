"""Agent execution layer.

This package is the stable boundary for the mechanics of running an Agent.
The legacy modules under :mod:`harness.agent`, :mod:`harness.session`, and
:mod:`harness.tools` remain available for compatibility during migration.
"""

from harness.agent.loop import Agent
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


__all__ = [
    "Agent",
    "InMemorySessionStore",
    "SessionStore",
    "ToolRuntime",
    "product_manager",
    "reset_product_manager",
    "request_header",
]
