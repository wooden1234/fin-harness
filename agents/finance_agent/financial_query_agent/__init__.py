"""financial_query_agent：结构化财务事实查询子 Agent。"""

from __future__ import annotations

_BUILT = False
_financial_query_agent = None


def build_financial_query_agent_graph():
    from agents.finance_agent.financial_query_agent.graph import (
        build_financial_query_agent_graph as _build,
    )

    return _build()


async def _run_financial_query_agent(state, config=None):
    """按 LangGraph 节点编排运行 financial_query_agent（测试兼容入口）。"""
    global _BUILT, _financial_query_agent
    if not _BUILT:
        _financial_query_agent = _build_subgraph()
        _BUILT = True
    return await _financial_query_agent.ainvoke(state, config)


def _build_subgraph() -> object:
    return build_financial_query_agent_graph().compile()


def __getattr__(name):
    global _BUILT, _financial_query_agent

    if name == "financial_query_agent":
        if not _BUILT:
            _financial_query_agent = _build_subgraph()
            _BUILT = True
        return _financial_query_agent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_financial_query_agent_graph",
    "financial_query_agent",
]
