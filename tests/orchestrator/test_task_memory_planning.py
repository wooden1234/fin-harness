from types import SimpleNamespace

import pytest

from agents.memory_recall import planning
from agents.orchestrator.contracts import TaskPlan, TaskSpec
from agents.orchestrator.graph import dispatch_wave, route_after_replan


def _plan(*tasks: TaskSpec) -> TaskPlan:
    return TaskPlan(plan_id="memory-plan", query="测试", tasks=list(tasks))


@pytest.mark.asyncio
async def test_memory_plan_uses_each_agent_static_whitelist():
    result = await planning.memory_plan_node(
        {
            "task_plan": _plan(
                TaskSpec(
                    task_id="general",
                    objective="普通回答",
                    agent_id="general_agent",
                ),
                TaskSpec(
                    task_id="finance",
                    objective="金融分析",
                    agent_id="finance_agent",
                ),
            ),
            "messages": [],
        }
    )

    general_keys = set(
        result["memory_requirements"]["general"]["memory_keys"]
    )
    finance_keys = set(
        result["memory_requirements"]["finance"]["memory_keys"]
    )
    assert "default_market" not in general_keys
    assert "default_market" in finance_keys
    assert general_keys < finance_keys
    assert (
        result["memory_requirements"]["general"]["semantic_memory_type"]
        == ""
    )
    assert (
        result["memory_requirements"]["finance"]["semantic_memory_type"]
        == ""
    )


@pytest.mark.asyncio
async def test_memory_plan_only_enables_explicit_semantic_history():
    result = await planning.memory_plan_node(
        {
            "task_plan": _plan(
                TaskSpec(
                    task_id="general",
                    objective="结合我之前的讨论",
                    agent_id="general_agent",
                    semantic_history=True,
                )
            ),
            "messages": [],
        }
    )

    requirement = result["memory_requirements"]["general"]
    assert requirement["semantic_memory_type"] == "episodic"
    assert requirement["semantic_query"] == "结合我之前的讨论"


@pytest.mark.asyncio
async def test_task_loader_keeps_parallel_agent_contexts_isolated(monkeypatch):
    calls: list[tuple[str, tuple[str, ...]]] = []

    async def fake_load_for_agent(
        *,
        tenant_id,
        user_id,
        agent_id,
        memory_keys,
        bypass_cache,
        audit_context,
    ):
        del tenant_id, user_id
        assert bypass_cache is False
        assert audit_context.task_id == "general"
        keys = tuple(memory_keys)
        calls.append((agent_id, keys))
        values = {
            key: {"owner": agent_id, "value": key}
            for key in keys
        }
        return SimpleNamespace(as_dict=lambda: values)

    monkeypatch.setattr(
        planning.MemoryLoader,
        "load_for_agent",
        fake_load_for_agent,
    )
    requirements = {
        "general": {
            "agent_id": "general_agent",
            "memory_keys": ["response_language"],
        },
        "finance": {
            "agent_id": "finance_agent",
            "memory_keys": ["response_language", "default_market"],
        },
    }
    runtime = SimpleNamespace(
        context=SimpleNamespace(tenant_id="tenant-1", user_id="7")
    )
    result = await planning.load_task_memories_node(
        {
            "memory_requirements": requirements,
            "turn_preferences": {"response_language": "en-US"},
        },
        runtime,
    )

    contexts = result["task_memory_context"]
    assert contexts["general"] == {"response_language": "en-US"}
    assert contexts["finance"]["response_language"] == "en-US"
    assert contexts["finance"]["default_market"]["owner"] == "finance_agent"
    assert "default_market" not in contexts["general"]
    contexts["finance"]["default_market"]["owner"] = "changed"
    assert contexts["general"] == {"response_language": "en-US"}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_replan_task_reuses_loaded_context_for_same_signature(monkeypatch):
    loader_calls = 0

    async def fake_load_for_agent(**_kwargs):
        nonlocal loader_calls
        loader_calls += 1
        return SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(
        planning.MemoryLoader,
        "load_for_agent",
        fake_load_for_agent,
    )
    requirement = {
        "agent_id": "finance_agent",
        "memory_keys": ["default_market"],
    }
    runtime = SimpleNamespace(
        context=SimpleNamespace(tenant_id="tenant-1", user_id="7")
    )
    result = await planning.load_task_memories_node(
        {
            "memory_requirements": {
                "finance": requirement,
                "finance:retry:1": requirement,
            },
            "task_memory_context": {
                "finance": {"default_market": "US"},
            },
        },
        runtime,
    )

    assert loader_calls == 0
    assert result["task_memory_context"]["finance:retry:1"] == {
        "default_market": "US"
    }
    assert route_after_replan({"next_action": "schedule"}) == "memory_plan"


def test_dispatch_wave_projects_only_current_task_memory():
    plan = _plan(
        TaskSpec(
            task_id="general",
            objective="普通回答",
            agent_id="general_agent",
        ),
        TaskSpec(
            task_id="finance",
            objective="金融分析",
            agent_id="finance_agent",
        ),
    )
    sends = dispatch_wave(
        {
            "task_plan": plan,
            "agent_results": [],
            "task_memory_context": {
                "general": {"response_language": "zh-CN"},
                "finance": {"default_market": "US"},
            },
        }
    )
    payloads = {
        send.arg["current_task"].task_id: send.arg
        for send in sends
    }

    assert payloads["general"]["current_task_memory_context"] == {
        "response_language": "zh-CN"
    }
    assert payloads["finance"]["current_task_memory_context"] == {
        "default_market": "US"
    }
    assert all("task_memory_context" not in payload for payload in payloads.values())
