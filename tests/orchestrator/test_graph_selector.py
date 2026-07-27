"""V1/V2 图选择和灰度稳定性测试。"""

import importlib

from app.core.config import settings
from agents.graph_selector import select_graph_version


def test_graph_mode_v1_can_be_selected(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", "v1")
    assert select_graph_version(conversation_id="c1", user_id="u1") == "v1"


def test_graph_mode_v2_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", "v2")
    assert select_graph_version(conversation_id="c1", user_id="u1") == "v2"


def test_graph_mode_defaults_to_v2(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", None)
    assert select_graph_version(conversation_id="c1", user_id="u1") == "v2"


def test_rollout_is_stable_for_same_identity(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", "rollout")
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_ROLLOUT_PERCENT", 50)
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_TENANT_ALLOWLIST", "")

    first = select_graph_version(
        conversation_id="c1",
        user_id="u1",
        tenant_id="tenant-a",
    )
    second = select_graph_version(
        conversation_id="c1",
        user_id="u1",
        tenant_id="tenant-a",
    )
    assert first == second


def test_rollout_allowlist_forces_v2(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", "rollout")
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_ROLLOUT_PERCENT", 0)
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_TENANT_ALLOWLIST", "tenant-a")

    assert (
        select_graph_version(
            conversation_id="c1",
            user_id="u1",
            tenant_id="tenant-a",
        )
        == "v2"
    )


def test_rollout_keeps_existing_integer_conversations_on_v1(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_GRAPH_MODE", "rollout")
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_ROLLOUT_PERCENT", 100)
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_TENANT_ALLOWLIST", "")
    monkeypatch.setattr(settings, "AGENT_GRAPH_V2_NEW_CONVERSATIONS_ONLY", True)

    assert select_graph_version(conversation_id=42, user_id="u1") == "v1"


def test_v1_and_v2_use_independent_graph_modules():
    v1_module = importlib.import_module("agent-v1.graph")
    from agents.orchestrator import graph as v2_module

    assert v1_module.__file__.endswith("agent-v1/graph.py")
    assert v1_module is not v2_module
    assert v1_module.get_graph(with_checkpointer=False) is not v2_module.get_orchestrator_graph(
        with_checkpointer=False
    )
