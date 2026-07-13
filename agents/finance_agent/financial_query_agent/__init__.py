"""financial_query_agent：结构化财务事实查询子 Agent。"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

_BUILT = False
_financial_query_agent = None


def _merge_steps(*step_lists: object) -> list[str]:
    merged: list[str] = []
    for step_list in step_lists:
        if not isinstance(step_list, list):
            continue
        merged.extend(str(item) for item in step_list)
    return merged


async def _run_financial_query_agent(state, config=None):
    """按 planner 结果分发到 predefined 或 text_to_sql。"""
    from agents.finance_agent.financial_query_agent.planner import (
        financial_query_planner,
    )
    from agents.finance_agent.financial_query_agent.workflows import (
        predefined_workflow,
        text_to_sql_workflow,
    )

    planner_updates = await financial_query_planner(state, config)
    route_name = str(planner_updates.get("financial_query_plan_route") or "")
    merged_state = {**state, **planner_updates}

    if route_name == "predefined":
        predefined_updates = await predefined_workflow(merged_state, config)
        if str(predefined_updates.get("financial_query_plan_route") or "") == "text_to_sql":
            fallback_state = {**merged_state, **predefined_updates}
            text_to_sql_updates = await text_to_sql_workflow(fallback_state, config)
            return {
                **planner_updates,
                **predefined_updates,
                **text_to_sql_updates,
                "steps": _merge_steps(
                    planner_updates.get("steps"),
                    predefined_updates.get("steps"),
                    text_to_sql_updates.get("steps"),
                ),
            }
        return {
            **planner_updates,
            **predefined_updates,
            "steps": _merge_steps(
                planner_updates.get("steps"),
                predefined_updates.get("steps"),
            ),
        }

    if route_name == "text_to_sql":
        text_to_sql_updates = await text_to_sql_workflow(merged_state, config)
        return {
            **planner_updates,
            **text_to_sql_updates,
            "steps": _merge_steps(
                planner_updates.get("steps"),
                text_to_sql_updates.get("steps"),
            ),
        }

    return planner_updates


def _build_subgraph() -> object:
    """惰性构建 runnable，在首次访问时组装。"""
    return RunnableLambda(_run_financial_query_agent)


def __getattr__(name):
    global _BUILT, _financial_query_agent

    if name == "financial_query_agent":
        if not _BUILT:
            _financial_query_agent = _build_subgraph()
            _BUILT = True
        return _financial_query_agent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["financial_query_agent"]
