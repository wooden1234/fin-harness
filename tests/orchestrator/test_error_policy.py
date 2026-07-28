"""V2 错误动作分类测试。"""

from agents.orchestrator.error_policy import classify_error


def test_transient_errors_retry() -> None:
    decision = classify_error("provider_timeout")

    assert decision.action == "retry"
    assert decision.retryable is True


def test_input_errors_clarify() -> None:
    decision = classify_error("market_query_plan_missing")

    assert decision.action == "clarify"


def test_provider_errors_fallback_inside_v2() -> None:
    decision = classify_error("source_tool_failed")

    assert decision.action == "fallback"


def test_permission_and_schema_errors_fail() -> None:
    assert classify_error("permission_denied").action == "fail"
    assert classify_error("schema_error").action == "fail"
