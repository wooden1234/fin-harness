"""financial_query_agent 子图编排。

planner → predefined / text_to_sql；predefined 失败时可降级到 text_to_sql。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.state import FinancialQueryRoute



async def planner_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    from agents.finance_agent.financial_query_agent.planner import (
        financial_query_planner,
    )

    return await financial_query_planner(state, config)


async def predefined_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    from agents.finance_agent.financial_query_agent.workflows import (
        predefined_workflow,
    )

    return await predefined_workflow(state, config)


async def text_to_sql_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    from agents.finance_agent.financial_query_agent.workflows import (
        text_to_sql_workflow,
    )

    return await text_to_sql_workflow(state, config)


def route_after_planner(state: FinAgentState) -> FinancialQueryRoute | str:
    route = str(state.get("financial_query_plan_route") or "")
    if route == "predefined":
        return "predefined"
    if route == "text_to_sql":
        return "text_to_sql"
    return END


def route_after_predefined(state: FinAgentState) -> FinancialQueryRoute | str:
    if str(state.get("financial_query_plan_route") or "") == "text_to_sql":
        return "text_to_sql"
    return END


def build_financial_query_agent_graph() -> StateGraph:
    """构建 financial_query_agent 路由图。"""
    builder = StateGraph(FinAgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("predefined", predefined_node)
    builder.add_node("text_to_sql", text_to_sql_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "predefined": "predefined",
            "text_to_sql": "text_to_sql",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "predefined",
        route_after_predefined,
        {
            "text_to_sql": "text_to_sql",
            END: END,
        },
    )
    builder.add_edge("text_to_sql", END)

    return builder


__all__ = [
    "build_financial_query_agent_graph",
    "planner_node",
    "predefined_node",
    "route_after_planner",
    "route_after_predefined",
    "text_to_sql_node",
]
