"""全架构合图（仅用于 Studio / Mermaid 可视化，不用于生产执行）。

各层关键节点按主链路串联成一张图，便于 LangSmith Studio 一次看清全貌。
（结构与真实条件路由略有简化，仅作架构总览。）
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

OverviewState = dict[str, Any]


async def _noop(state: OverviewState) -> OverviewState:
    return state


def build_combined_overview_graph() -> StateGraph:
    """串联全架构节点，单路径无 fan-in，确保 Studio / draw_mermaid 可渲染。"""
    builder = StateGraph(OverviewState)

    chain = (
        "guardrails",
        "context_compressor",
        "supervisor",
        "risk_triage",
        "plan_tasks",
        "validate_plan",
        "resolve_evidence",
        "dispatch_workers",
        "faq_agent",
        "pdf_agent",
        "financial_query_agent",
        "fq_planner",
        "fq_predefined",
        "pd_init",
        "pd_select_tool",
        "pd_semantic",
        "pd_resolve",
        "pd_execute",
        "pd_format",
        "fq_text_to_sql",
        "t2s_prepare",
        "t2s_generate",
        "t2s_validate",
        "t2s_db_verify",
        "t2s_format",
        "web_search_agent",
        "join",
        "summarize",
        "general_agent",
        "final_answer",
    )
    for name in chain:
        builder.add_node(name, _noop)

    builder.add_edge(START, chain[0])
    for left, right in zip(chain, chain[1:]):
        builder.add_edge(left, right)
    builder.add_edge(chain[-1], END)

    return builder


__all__ = ["build_combined_overview_graph"]
