"""主图编译：START → guardrails → context_compressor → supervisor → risk_triage → plan_agent → final_answer → END。

plan_agent 作为独立子图（planner → workers → join → summarize），主图只感知一个节点。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.states import FinAgentInput, FinAgentState
from agents.context_compressor import compress_context
from agents.final_answer import final_answer_node
from agents.general_agent.node import general_agent
from agents.finance_agent import finance_agent as _finance_agent_graph
from agents.guardrails import guardrails_edge, guardrails_node
from agents.risk_triage import risk_triage_edge, risk_triage_node
from agents.supervisor import analyze_and_route_query, route_query

_compiled_graph = None


def build_graph() -> StateGraph:
    """构建未编译的 StateGraph（便于单测与 export）。"""
    builder = StateGraph(FinAgentState, input_schema=FinAgentInput)

    builder.add_node("guardrails", guardrails_node)
    builder.add_node("context_compressor", compress_context)
    builder.add_node("supervisor", analyze_and_route_query)
    builder.add_node("risk_triage", risk_triage_node)
    builder.add_node("general_agent", general_agent)
    builder.add_node("plan_agent", _finance_agent_graph)
    builder.add_node("final_answer", final_answer_node)

    # Layer 1: START → guardrails
    builder.add_edge(START, "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        guardrails_edge,
        {
            "context_compressor": "context_compressor",
            "final_answer": "final_answer",
        },
    )

    # Layer 2: context_compressor → supervisor
    builder.add_edge("context_compressor", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "general_agent": "general_agent",
            "risk_triage": "risk_triage",
            "error_handler": "final_answer",
        },
    )

    # Layer 3: risk_triage → plan_agent / END
    builder.add_conditional_edges(
        "risk_triage",
        risk_triage_edge,
        {
            "plan_agent": "plan_agent",
            "__end__": END,
        },
    )

    # Layer 4: 汇聚 → final_answer → END
    builder.add_edge("general_agent", "final_answer")
    builder.add_edge("plan_agent", "final_answer")
    builder.add_edge("final_answer", END)

    return builder


def compile_graph(checkpointer: BaseCheckpointSaver | None):
    return build_graph().compile(checkpointer=checkpointer)


def reset_graph_cache() -> None:
    global _compiled_graph
    _compiled_graph = None


def get_graph(*, with_checkpointer: bool = True):
    """返回编译后的主图。

    - ``with_checkpointer=False``：无持久化（export / 结构单测）
    - ``with_checkpointer=True``：使用 ``init_checkpoint()`` 后的 saver（Postgres 或 Memory）
    """
    global _compiled_graph

    if not with_checkpointer:
        return compile_graph(None)

    if _compiled_graph is None:
        from agents.checkpoint import get_checkpointer

        _compiled_graph = compile_graph(get_checkpointer())
    return _compiled_graph


def get_graph_with_memory():
    """测试辅助：独立 MemorySaver，不依赖全局 init。"""
    return compile_graph(MemorySaver())
