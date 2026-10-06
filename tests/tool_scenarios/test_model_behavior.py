"""真实模型 + 假工具行为评测。只有显式开关才会调用模型API。"""

from __future__ import annotations

import os
from typing import Any

import pytest

from evals.tool_dataset import load_tool_cases
from tests.tool_scenarios.model_behavior_driver import ModelBehaviorDriver


CALIBRATION_IDS = {
    "tool-001", "tool-004", "tool-005", "tool-007", "tool-019",
    "tool-021", "tool-023", "tool-025", "tool-029", "tool-031",
    "tool-032", "tool-033", "tool-041", "tool-050", "tool-053",
    "tool-056", "tool-061", "tool-065", "tool-067", "tool-070",
}


def pytest_generate_tests(metafunc: Any) -> None:
    if "model_case" not in metafunc.fixturenames:
        return
    config = metafunc.config
    cases = [
        case for case in load_tool_cases("full")
        if case["metadata"]["execution_mode"] == "agent"
    ]
    selected_ids = set(config.getoption("--model-case-id") or [])
    if selected_ids:
        cases = [case for case in cases if case["id"] in selected_ids]
    elif not config.getoption("--model-all"):
        cases = [case for case in cases if case["id"] in CALIBRATION_IDS]
    repetitions = max(1, int(config.getoption("--model-repetitions")))
    params = [(case, repetition) for case in cases for repetition in range(1, repetitions + 1)]
    metafunc.parametrize(
        ("model_case", "repetition"),
        params,
        ids=[f"{case['id']}-r{repetition}" for case, repetition in params],
    )


@pytest.mark.model_behavior
@pytest.mark.asyncio(loop_scope="session")
async def test_real_model_with_fake_tools(
    model_case: dict[str, Any],
    repetition: int,
    record_property: Any,
    request: Any,
) -> None:
    if not request.config.getoption("--run-model-behavior"):
        pytest.skip("需要显式传入 --run-model-behavior，避免意外消耗模型额度")
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("未配置 DEEPSEEK_API_KEY")

    observation = await ModelBehaviorDriver().run(model_case)
    record_property("tool_case_id", model_case["id"])
    record_property("tool_title", model_case["metadata"]["title"])
    record_property("tool_bucket", model_case["metadata"]["bucket"])
    record_property("tool_execution_mode", "model_behavior")
    record_property("tool_profile", "model_behavior")
    record_property("tool_repetition", repetition)
    record_property("tool_model", observation.model)
    record_property("tool_names", observation.tool_names)
    record_property("tool_call_count", len(observation.tool_names))
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
            "duplicate_requests": 0,
            "unbudgeted_subcalls": 0,
            "first_attempt_successes": observation.first_attempt_successes,
            "judge_reason": observation.judge_reason,
        },
    )
    assert "failed" not in observation.assertions.values(), {
        **observation.assertions,
        "judge_reason": observation.judge_reason,
    }
