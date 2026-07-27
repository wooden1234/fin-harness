"""V1 主图：固定根路由 + Finance 领域子图。

V1 与 V2 使用独立的编译缓存和入口。领域节点仍复用共享的工具、合规和
运行时上下文，但不会复用另一版本的根图或 checkpoint 线程。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from agents.context_compressor import compress_context
from agents.final_answer import final_answer_node
from agents.finance_agent import finance_agent as _finance_agent_graph
from agents.general_agent.node import general_agent
from agents.guardrails import guardrails_edge, guardrails_node
from agents.init_turn import init_turn_node
from agents.memory_recall import memory_recall_node
from agents.query_rewrite import query_rewrite_node, route_after_query_rewrite
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentInput, FinAgentState
from agents.stock_screening_agent import stock_screening_agent
from agents.supervisor import analyze_and_route_query, route_query

_compiled_graph = None


def build_graph() -> StateGraph:
    """构建未编译的 V1 StateGraph。"""
    builder = StateGraph(
        FinAgentState,
        input_schema=FinAgentInput,
        context_schema=AgentRuntimeContext,
    )

    builder.add_node("init_turn", init_turn_node)
    builder.add_node("memory_recall", memory_recall_node)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("context_compressor", compress_context)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("supervisor", analyze_and_route_query)
    builder.add_node("general_agent", general_agent)
    builder.add_node("stock_screening_agent", stock_screening_agent)
    builder.add_node("plan_agent", _finance_agent_graph)
    builder.add_node("final_answer", final_answer_node)

    builder.add_edge(START, "init_turn")
    builder.add_edge("init_turn", "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        guardrails_edge,
        {"memory_recall": "memory_recall", "final_answer": "final_answer"},
    )
    builder.add_edge("memory_recall", "context_compressor")
    builder.add_edge("context_compressor", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "general_agent": "general_agent",
            "plan_agent": "plan_agent",
            "stock_screening_agent": "stock_screening_agent",
            "query_rewrite": "query_rewrite",
            "final_answer": "final_answer",
            "error_handler": "final_answer",
        },
    )
    builder.add_conditional_edges(
        "query_rewrite",
        route_after_query_rewrite,
        {"supervisor": "supervisor", "final_answer": "final_answer"},
    )
    builder.add_edge("general_agent", "final_answer")
    builder.add_edge("plan_agent", "final_answer")
    builder.add_edge("stock_screening_agent", "final_answer")
    builder.add_edge("final_answer", END)
    return builder


def compile_graph(
    checkpointer: BaseCheckpointSaver | None,
    store: BaseStore | None = None,
):
    return build_graph().compile(checkpointer=checkpointer, store=store)


def reset_graph_cache() -> None:
    global _compiled_graph
    _compiled_graph = None


def get_graph(*, with_checkpointer: bool = True):
    """返回 V1 编译图；V2 不使用此缓存。"""
    global _compiled_graph
    if not with_checkpointer:
        return compile_graph(None)
    if _compiled_graph is None:
        from agents.checkpoint import get_checkpointer
        from app.services.memory.memory_store import get_memory_store

        _compiled_graph = compile_graph(get_checkpointer(), get_memory_store())
    return _compiled_graph


def get_graph_with_memory():
    """测试辅助：使用独立 MemorySaver 构建 V1 图。"""
    return compile_graph(MemorySaver())


__all__ = [
    "build_graph",
    "compile_graph",
    "get_graph",
    "get_graph_with_memory",
    "reset_graph_cache",
]
