"""financial_query_agent：结构化财务事实查询子 Agent。"""

from __future__ import annotations

import importlib
import sys

_BUILT = False
_financial_query_agent = None


def build_financial_query_agent_graph():
    from agents.finance_agent.financial_query_agent.graph import (
        build_financial_query_agent_graph as _build,
    )

    return _build()


async def _run_financial_query_agent(state, config=None):
    """按与子图相同的路由顺序运行，供单元测试和脚本直接调用。"""
    from agents.finance_agent.financial_query_agent.planner import (
        financial_query_planner,
    )
    from agents.finance_agent.financial_query_agent.workflows import (
        predefined_workflow,
        text_to_sql_workflow,
    )

    merged = dict(state)

    def apply(updates):
        for key, value in dict(updates or {}).items():
            if key in {"messages", "task_results", "citations", "steps"}:
                merged[key] = [*list(merged.get(key) or []), *list(value or [])]
            else:
                merged[key] = value

    apply(await financial_query_planner(merged, config))
    route = str(merged.get("financial_query_plan_route") or "")
    if route == "predefined":
        apply(await predefined_workflow(merged, config))
        route = str(merged.get("financial_query_plan_route") or "")
    if route == "text_to_sql":
        apply(await text_to_sql_workflow(merged, config))
    return merged


def _build_subgraph() -> object:
    return build_financial_query_agent_graph().compile()


def __getattr__(name):
    global _BUILT, _financial_query_agent

    if name == "financial_query_agent":
        if not _BUILT:
            _financial_query_agent = _build_subgraph()
            _BUILT = True
        return _financial_query_agent

    if name in {"graph", "planner", "services", "text_to_sql", "workflows"}:
        # 兼容历史 app.agents shim 与 canonical 包混用时的属性覆盖。
        canonical_name = f"agents.finance_agent.financial_query_agent.{name}"
        return sys.modules.get(canonical_name) or importlib.import_module(
            canonical_name
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_financial_query_agent_graph",
    "financial_query_agent",
]
