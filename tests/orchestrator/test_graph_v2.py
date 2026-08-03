"""根 Orchestrator 图结构和当前执行路径测试。"""

from agents.orchestrator.graph import build_orchestrator_graph


def test_graph_compiles_with_current_root_path():
    graph = build_orchestrator_graph().compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    for node in (
        "init_turn",
        "guardrails",
        "memory_action",
        "memory_recall",
        "classify_execution_lane",
        "resolve_execution_lane",
        "context_compressor",
        "general_agent",
        "main_deep_agent",
        "evidence_quality_gate",
        "final_answer",
        "post_turn_memory",
    ):
        assert node in graph.nodes

    assert ("__start__", "init_turn") in edges
    assert ("init_turn", "guardrails") in edges
    assert ("guardrails", "memory_action") in edges
    assert ("memory_action", "memory_recall") in edges
    assert ("memory_recall", "classify_execution_lane") in edges
    assert ("classify_execution_lane", "context_compressor") in edges
    assert ("classify_execution_lane", "resolve_execution_lane") in edges
    assert ("resolve_execution_lane", "context_compressor") in edges
    assert ("context_compressor", "general_agent") in edges
    assert ("context_compressor", "main_deep_agent") in edges
    assert ("general_agent", "final_answer") in edges
    assert ("main_deep_agent", "evidence_quality_gate") in edges
    assert ("evidence_quality_gate", "final_answer") in edges
    assert ("final_answer", "post_turn_memory") in edges
    assert ("post_turn_memory", "__end__") in edges
