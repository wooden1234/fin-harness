"""V2 deadline、任务超时、并发限制和取消传播测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import time

import pytest

from agents.orchestrator.contracts import (
    AgentResult,
    QualityReport,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.graph import (
    build_plan,
    evaluate_results,
    execute_task,
    prepare_wave,
    replan,
)
from agents.runtime_context import AgentRuntimeContext


def _runtime(context: AgentRuntimeContext):
    return SimpleNamespace(context=context)


async def test_execute_task_returns_standard_timeout_result(
    monkeypatch,
) -> None:
    async def slow_invoke(task, **kwargs):
        del task, kwargs
        await asyncio.sleep(0.05)

    monkeypatch.setattr("agents.orchestrator.graph.invoke_agent", slow_invoke)
    monkeypatch.setattr(
        "agents.orchestrator.graph.settings.AGENT_V2_AGENT_TIMEOUT_SEC",
        0.01,
    )

    update = await execute_task(
        {
            "current_task": TaskSpec(
                task_id="finance",
                objective="查询营收",
                agent_id="finance_agent",
            )
        },
        runtime=_runtime(AgentRuntimeContext(max_concurrency=1)),
    )

    result = update["agent_results"][0]
    assert result.status == "failed"
    assert result.error_code == "task_timeout"


async def test_execute_task_respects_expired_run_deadline() -> None:
    context = AgentRuntimeContext(
        deadline_monotonic=time.monotonic() - 1,
        max_concurrency=1,
    )

    update = await execute_task(
        {
            "current_task": TaskSpec(
                task_id="finance",
                objective="查询营收",
                agent_id="finance_agent",
            )
        },
        runtime=_runtime(context),
    )

    result = update["agent_results"][0]
    assert result.status == "failed"
    assert result.error_code == "run_hard_deadline_exceeded"


async def test_execute_task_uses_soft_deadline_for_graceful_degrade(
    monkeypatch,
) -> None:
    async def slow_invoke(task, **kwargs):
        del task, kwargs
        await asyncio.sleep(0.05)

    monkeypatch.setattr("agents.orchestrator.graph.invoke_agent", slow_invoke)
    context = AgentRuntimeContext(max_concurrency=1)
    context.configure_budget(
        complexity="simple",
        soft_seconds=0.01,
        hard_seconds=1,
        unit_timeouts={"agent": 0.5},
    )

    update = await execute_task(
        {
            "current_task": TaskSpec(
                task_id="general",
                objective="你好",
                agent_id="general_agent",
            )
        },
        runtime=_runtime(context),
    )
    result = update["agent_results"][0]
    route_update = await evaluate_results({"agent_results": [result]})

    assert result.error_code == "run_soft_deadline_exceeded"
    assert route_update["execution_status"] == "soft_timeout"
    assert route_update["next_action"] == "synthesize"


async def test_execute_task_limits_parallel_invocations(monkeypatch) -> None:
    active = 0
    max_active = 0

    async def tracked_invoke(task, **kwargs):
        nonlocal active, max_active
        del kwargs
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="completed",
        )

    monkeypatch.setattr("agents.orchestrator.graph.invoke_agent", tracked_invoke)
    runtime = _runtime(AgentRuntimeContext(max_concurrency=1))
    states = [
        {
            "current_task": TaskSpec(
                task_id=f"general-{index}",
                objective="你好",
                agent_id="general_agent",
            )
        }
        for index in range(3)
    ]

    await asyncio.gather(
        *(execute_task(state, runtime=runtime) for state in states)
    )

    assert max_active == 1


async def test_execute_task_propagates_cancellation(monkeypatch) -> None:
    async def cancelled_invoke(task, **kwargs):
        del task, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "agents.orchestrator.graph.invoke_agent",
        cancelled_invoke,
    )

    with pytest.raises(asyncio.CancelledError):
        await execute_task(
            {
                "current_task": TaskSpec(
                    task_id="general",
                    objective="你好",
                    agent_id="general_agent",
                )
            },
            runtime=_runtime(AgentRuntimeContext(max_concurrency=1)),
        )


def test_runtime_context_tracks_remaining_deadline() -> None:
    context = AgentRuntimeContext(
        deadline_monotonic=time.monotonic() + 1,
        max_concurrency=0,
    )

    remaining = context.remaining_seconds()
    assert remaining is not None
    assert 0 < remaining <= 1
    assert context.max_concurrency == 1


async def test_build_plan_configures_budget_by_request_complexity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agents.orchestrator.graph.settings.AGENT_V2_SIMPLE_SOFT_DEADLINE_SEC",
        2.0,
    )
    monkeypatch.setattr(
        "agents.orchestrator.graph.settings.AGENT_V2_SIMPLE_HARD_DEADLINE_SEC",
        4.0,
    )
    context = AgentRuntimeContext(max_concurrency=1)

    await build_plan(
        {
            "request_profile": RequestProfile(
                original_query="你好",
                complexity="simple",
            )
        },
        runtime=_runtime(context),
    )

    assert context.request_complexity == "simple"
    assert 0 < context.soft_remaining_seconds() <= 2
    assert 0 < context.remaining_seconds() <= 4
    assert context.unit_timeouts["tool_skill"] > 0


async def test_soft_deadline_stops_new_wave_and_replan() -> None:
    context = AgentRuntimeContext(
        started_monotonic=time.monotonic() - 2,
        max_concurrency=1,
    )
    context.configure_budget(
        complexity="simple",
        soft_seconds=1,
        hard_seconds=10,
        unit_timeouts={"agent": 3},
    )
    task = TaskSpec(
        task_id="general",
        objective="你好",
        agent_id="general_agent",
    )
    state = {
        "task_plan": TaskPlan(plan_id="plan", query="你好", tasks=[task]),
        "agent_results": [],
        "quality_report": QualityReport(
            passed=False,
            failed_task_ids=["general"],
        ),
    }

    wave_update = await prepare_wave(state, runtime=_runtime(context))
    replan_update = await replan(state, runtime=_runtime(context))

    assert wave_update["execution_status"] == "soft_timeout"
    assert wave_update["active_task_ids"] == []
    assert replan_update["execution_status"] == "soft_timeout"
    assert replan_update["next_action"] == "synthesize"


def test_tool_skill_budget_is_limited_by_run_hard_deadline() -> None:
    context = AgentRuntimeContext(
        deadline_monotonic=time.monotonic() + 1,
        max_concurrency=1,
    )
    context.configure_budget(
        complexity="compound",
        soft_seconds=5,
        hard_seconds=20,
        unit_timeouts={"tool_skill": 10},
    )

    timeout_seconds, timeout_limit = context.execution_timeout_for(
        "tool_skill",
        default_seconds=10,
    )

    assert 0 < timeout_seconds <= 1
    assert timeout_limit == "hard"
