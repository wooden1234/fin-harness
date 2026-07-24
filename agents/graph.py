"""主图编译：START → init_turn → guardrails → memory_recall → context_compressor → supervisor → 按需 query_rewrite → plan_agent → final_answer → END。

plan_agent 作为独立子图（planner → workers → join → summarize），主图只感知一个节点。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.states import FinAgentInput, FinAgentState
from agents.runtime_context import AgentRuntimeContext
from agents.init_turn import init_turn_node
from agents.memory_recall import memory_recall_node
from langgraph.store.base import BaseStore
from agents.context_compressor import compress_context
from agents.final_answer import final_answer_node
from agents.general_agent.node import general_agent
from agents.finance_agent import finance_agent as _finance_agent_graph
from agents.guardrails import guardrails_edge, guardrails_node
from agents.query_rewrite import query_rewrite_node, route_after_query_rewrite
from agents.supervisor import analyze_and_route_query, route_query

_compiled_graph = None


def build_graph() -> StateGraph:
    """构建未编译的 StateGraph（便于单测与 export）。"""
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
    builder.add_node("plan_agent", _finance_agent_graph)
    builder.add_node("final_answer", final_answer_node)

    # Layer 1: START → init_turn → guardrails → memory_recall
    builder.add_edge(START, "init_turn")
    builder.add_edge("init_turn", "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        guardrails_edge,
        {
            "memory_recall": "memory_recall",
            "final_answer": "final_answer",
        },
    )

    # Layer 2: memory_recall → context_compressor → supervisor（按需改写一次）
    builder.add_edge("memory_recall", "context_compressor")
    builder.add_edge("context_compressor", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "general_agent": "general_agent",
            "plan_agent": "plan_agent",
            "query_rewrite": "query_rewrite",
            "final_answer": "final_answer",
            "error_handler": "final_answer",
        },
    )
    builder.add_conditional_edges(
        "query_rewrite",
        route_after_query_rewrite,
        {
            "supervisor": "supervisor",
            "final_answer": "final_answer",
        },
    )

    # Layer 3: 汇聚 → final_answer → END
    builder.add_edge("general_agent", "final_answer")
    builder.add_edge("plan_agent", "final_answer")
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
    """返回编译后的主图。

    - ``with_checkpointer=False``：无持久化（export / 结构单测）
    - ``with_checkpointer=True``：使用 ``init_checkpoint()`` 后的 saver（Postgres 或 Memory）
    """
    global _compiled_graph

    if not with_checkpointer:
        return compile_graph(None)

    if _compiled_graph is None:
        from agents.checkpoint import get_checkpointer
        from app.services.memory.memory_store import get_memory_store

        _compiled_graph = compile_graph(get_checkpointer(), get_memory_store())
    return _compiled_graph


def get_graph_with_memory():
    """测试辅助：独立 MemorySaver，不依赖全局 init。"""
    return compile_graph(MemorySaver())
