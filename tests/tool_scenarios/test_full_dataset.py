"""100 条工具评测集的参数化 Scheduler/Pipeline 执行入口。"""

from __future__ import annotations

from typing import Any

import pytest

from evals.tool_dataset import load_tool_cases
from tests.tool_scenarios.scenario_driver import ScenarioDriver


CASES = load_tool_cases("full")


@pytest.mark.full
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_full_tool_scenario(case: dict[str, Any], record_property: Any) -> None:
    observation = await ScenarioDriver().run(case)
    record_property("tool_case_id", case["id"])
    record_property("tool_title", case["metadata"]["title"])
    record_property("tool_bucket", case["metadata"]["bucket"])
    record_property("tool_execution_mode", case["metadata"]["execution_mode"])
    record_property("tool_names", observation.tool_names)
    record_property("tool_call_count", observation.logical_calls)
    record_property("tool_answer", observation.answer)
    record_property("tool_finish_reason", observation.finish_reason)
    record_property("tool_assertions", observation.assertions)
    record_property(
        "tool_metrics",
        {
            "logical_calls": observation.logical_calls,
            "actual_requests": observation.actual_requests,
            "retries": observation.retries,
            "retry_eligible_calls": observation.retry_eligible_calls,
            "bounded_recovery_successes": observation.bounded_recovery_successes,
            "duplicate_requests": observation.duplicate_requests,
            "unbudgeted_subcalls": observation.unbudgeted_subcalls,
            "first_attempt_successes": observation.first_attempt_successes,
        },
    )
    evaluated = [value for value in observation.assertions.values() if value != "not_evaluated"]
    assert evaluated, "场景没有任何可执行断言"
    assert "failed" not in evaluated, observation.assertions
