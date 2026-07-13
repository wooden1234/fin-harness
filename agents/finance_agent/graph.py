"""finance_agent 子图构建：plan → validate/repair → dispatch → workers → join → summarize。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from agents.states import FinAgentState
from agents.finance_agent.planner import (
    dispatch_workers_node,
    plan_tasks_node,
    repair_plan_node,
    route_after_dispatch_workers,
    route_after_retrieval_worker,
    route_after_validate_plan,
    validate_plan_node,
)
from agents.finance_agent.faq_agent import faq_agent
from agents.finance_agent.pdf_agent import pdf_agent
from agents.finance_agent.financial_query_agent import financial_query_agent
from agents.finance_agent.web_search_agent import web_search_agent
from agents.finance_agent.join import join_node, route_after_join
from agents.finance_agent.summarize import summarize_node
from agents.finance_agent.workers import isolate_worker_node


def build_finance_agent_subgraph() -> StateGraph:
    """构建 finance_agent 子图：显式 Supervisor 子流程 + Worker 扇出/汇聚。"""
    builder = StateGraph(FinAgentState)

    builder.add_node("plan_tasks", plan_tasks_node)
    builder.add_node("validate_plan", validate_plan_node)
    builder.add_node("repair_plan", repair_plan_node)
    builder.add_node("dispatch_workers", dispatch_workers_node)
    builder.add_node("faq_agent", isolate_worker_node(faq_agent))
    builder.add_node("pdf_agent", isolate_worker_node(pdf_agent))
    builder.add_node(
        "financial_query_agent",
        isolate_worker_node(financial_query_agent),
    )
    builder.add_node("web_search_agent", isolate_worker_node(web_search_agent))
    builder.add_node("join", join_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "plan_tasks")
    builder.add_edge("plan_tasks", "validate_plan")
    builder.add_conditional_edges(
        "validate_plan",
        route_after_validate_plan,
        {
            "repair_plan": "repair_plan",
            "dispatch_workers": "dispatch_workers",
        },
    )
    builder.add_edge("repair_plan", "dispatch_workers")
    builder.add_conditional_edges("dispatch_workers", route_after_dispatch_workers)
    builder.add_conditional_edges(
        "faq_agent",
        route_after_retrieval_worker,
        {
            "web_search_agent": "web_search_agent",
            "join": "join",
        },
    )
    builder.add_conditional_edges(
        "pdf_agent",
        route_after_retrieval_worker,
        {
            "web_search_agent": "web_search_agent",
            "join": "join",
        },
    )
    builder.add_edge("financial_query_agent", "join")
    builder.add_edge("web_search_agent", "join")
    builder.add_conditional_edges(
        "join",
        route_after_join,
        {
            "summarize": "summarize",
            END: END,
        },
    )
    builder.add_edge("summarize", END)

    return builder


__all__ = ["build_finance_agent_subgraph"]
