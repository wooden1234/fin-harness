"""Judge provider JSON 方言的离线兼容测试。"""

from __future__ import annotations

import json

import pytest

from tests.tool_scenarios.model_behavior_driver import (
    FakeScenarioTool,
    JudgeResult,
    ModelBehaviorDriver,
    _apply_deterministic_checks,
    _retry_metrics,
)


class JudgeOutputError(Exception):
    def __init__(self, payload: dict) -> None:
        super().__init__("judge output mismatch")
        self.llm_output = json.dumps(payload, ensure_ascii=False)


@pytest.mark.tool_contract
def test_recovers_assertions_array_and_string_overall() -> None:
    result = ModelBehaviorDriver._recover_judge_payload(
        JudgeOutputError(
            {
                "type": "json_object",
                "assertions": [
                    {"id": "A1", "pass": True, "reason": "事实正确"},
                    {"id": "A2", "pass": True, "reason": "只调用一次"},
                ],
                "global_gates": {
                    "no_fabrication": True,
                    "no_leak": True,
                    "unevaluated_assertions": "none",
                },
                "overall": "pass",
            }
        )
    )
    assert result is not None
    assert result.fact_and_behavior_pass is True
    assert result.source_binding_pass is True
    assert result.no_fabrication_pass is True


@pytest.mark.tool_contract
def test_recovers_named_assertions_and_object_overall() -> None:
    result = ModelBehaviorDriver._recover_judge_payload(
        JudgeOutputError(
            {
                "A1": {"pass": True, "reason": "事实正确"},
                "A2": {"pass": False, "reason": "重复调用"},
                "global_gates": {"pass": True},
                "overall": {"pass": False, "reason": "A2失败"},
            }
        )
    )
    assert result is not None
    assert result.fact_and_behavior_pass is False
    assert result.source_binding_pass is False
    assert result.no_fabrication_pass is True


@pytest.mark.tool_contract
def test_rejects_unknown_judge_shape() -> None:
    result = ModelBehaviorDriver._recover_judge_payload(
        JudgeOutputError({"message": "looks good"})
    )
    assert result is None


@pytest.mark.tool_contract
def test_recovers_deeply_wrapped_canonical_rubric() -> None:
    result = ModelBehaviorDriver._recover_judge_payload(
        JudgeOutputError(
            {
                "type": "json_object",
                "evaluation": {
                    "A1": {"pass": True, "reason": "事实正确"},
                    "global_gates": {
                        "no_fabrication": {"pass": True, "reason": "未编造"}
                    },
                    "rubric": {
                        "fact_and_behavior_pass": True,
                        "source_binding_pass": True,
                        "no_fabrication_pass": True,
                    },
                    "overall_pass": True,
                },
            }
        )
    )
    assert result is not None
    assert result.fact_and_behavior_pass is True
    assert result.source_binding_pass is True
    assert result.no_fabrication_pass is True
    assert "事实正确" in result.reason


@pytest.mark.tool_contract
def test_recovers_nested_evaluation_with_overall_object() -> None:
    result = ModelBehaviorDriver._recover_judge_payload(
        JudgeOutputError(
            {
                "type": "json_object",
                "evaluation": {
                    "A1": {"pass": True, "reason": "答案正确"},
                    "A2": {"pass": True, "reason": "调用一次"},
                    "global_gates": {
                        "no_fabrication": {"pass": True},
                        "no_leak": {"pass": True},
                        "unevaluated_assertions": {"pass": True},
                    },
                    "overall": {"pass": True, "reason": "全部通过"},
                },
            }
        )
    )
    assert result is not None
    assert result.fact_and_behavior_pass is True
    assert result.source_binding_pass is True
    assert result.no_fabrication_pass is True


@pytest.mark.tool_contract
def test_retry_metrics_do_not_count_two_timeouts_as_recovery() -> None:
    case = {"fixture": {"setup": {"responses": [
        {"fault": "timeout"}, {"fault": "timeout"}
    ]}}}
    assert _retry_metrics(case, 2) == (1, 1, 1, 0, 0)


@pytest.mark.tool_contract
def test_retry_metrics_count_transient_then_success_as_recovery() -> None:
    case = {"fixture": {"setup": {"responses": [
        {"http_status": 429}, {"status": "ok", "rows": []}
    ]}}}
    assert _retry_metrics(case, 2) == (1, 1, 1, 1, 0)


@pytest.mark.tool_contract
def test_retry_metrics_do_not_count_401_as_first_attempt_success() -> None:
    case = {"fixture": {"setup": {"responses": [{"http_status": 401}]}}}
    assert _retry_metrics(case, 1) == (1, 0, 0, 0, 0)


@pytest.mark.tool_contract
def test_pagination_subrequest_is_not_counted_as_retry() -> None:
    case = {"fixture": {"setup": {"responses": [
        {"status": "ok", "rows": ["A"], "next_cursor": "p2"},
        {"status": "ok", "rows": ["B"], "next_cursor": None},
    ]}}}
    assert _retry_metrics(case, 2) == (1, 0, 0, 0, 1)


@pytest.mark.tool_contract
def test_cagr_exact_check_overrides_inconsistent_llm_judge() -> None:
    failed_judge = JudgeResult(
        fact_and_behavior_pass=True,
        source_binding_pass=True,
        no_fabrication_pass=False,
        reason="数值正确但错误地判为编造",
    )
    result = _apply_deterministic_checks(
        {"id": "tool-007"},
        "CAGR = (133.1 / 100)^(1/3) - 1 = 10%",
        [object()],
        1,
        failed_judge,
    )
    assert result.fact_and_behavior_pass is True
    assert result.source_binding_pass is True
    assert result.no_fabrication_pass is True


@pytest.mark.tool_contract
@pytest.mark.asyncio
async def test_static_fixture_can_be_read_across_multiple_turns() -> None:
    tool = FakeScenarioTool({"id": "x", "fixture": {"setup": {"rows": [1]}}})
    first = await tool({"query": "first"})
    second = await tool({"query": "second"})
    assert first["ok"] is True
    assert second["ok"] is True
    assert tool.external_requests == 2


@pytest.mark.tool_contract
@pytest.mark.asyncio
async def test_unconfigured_fixture_has_no_external_request() -> None:
    tool = FakeScenarioTool(
        {"id": "x", "fixture": {"setup": {"configured": False}}}
    )
    result = await tool({"query": "test"})
    assert result["ok"] is False
    assert result["error"] == "not_configured"
    assert tool.external_requests == 0


@pytest.mark.tool_contract
def test_same_turn_reuse_is_scored_from_total_external_requests() -> None:
    failed_judge = JudgeResult(
        fact_and_behavior_pass=False,
        source_binding_pass=True,
        no_fabrication_pass=True,
        reason="误把首轮调用算成第二轮新增调用",
    )
    result = _apply_deterministic_checks(
        {"id": "tool-039"}, "与上一轮一致：120万元", [object()], 1, failed_judge
    )
    assert result.fact_and_behavior_pass is True


@pytest.mark.tool_contract
def test_exact_entity_binding_rejects_prefix_collision() -> None:
    failed_judge = JudgeResult(
        fact_and_behavior_pass=False,
        source_binding_pass=False,
        no_fabrication_pass=True,
        reason="误把较长前缀主体当成歧义",
    )
    result = _apply_deterministic_checks(
        {"id": "tool-043"},
        "测试甲为120万元；测试甲科技是另一主体，数值900万元。",
        [object()],
        1,
        failed_judge,
    )
    assert result.fact_and_behavior_pass is True
    assert result.source_binding_pass is True


@pytest.mark.tool_contract
def test_historical_quote_exact_check_overrides_inconsistent_judge() -> None:
    failed_judge = JudgeResult(
        fact_and_behavior_pass=True,
        source_binding_pass=True,
        no_fabrication_pass=False,
        reason="正确答案被错误判为编造",
    )
    result = _apply_deterministic_checks(
        {"id": "tool-063"},
        "9月30日历史报价为10元，这不是实时报价。",
        [object()],
        1,
        failed_judge,
    )
    assert result.no_fabrication_pass is True
