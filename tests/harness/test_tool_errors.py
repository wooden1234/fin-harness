from datetime import datetime, timezone

from harness.session.types import SessionEvent
from harness.tools.error_policy import decide_after_tools
from harness.tools.errors import (
    ToolErrorAction,
    ToolErrorClass,
    classify,
    enrich_tool_result,
    error_result,
)
from harness.tools.retry_policy import USER_UNAVAILABLE_HINT


def _event(*, event_type: str = "tool/result", data: dict, turn: int = 1) -> SessionEvent:
    return SessionEvent(
        seq=1,
        event_id="e",
        event_type=event_type,
        schema_version=1,
        run_id="r",
        turn=turn,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="append",
        source_event_seqs=(),
        visibility="public",
        data=data,
        created_at=datetime.now(timezone.utc),
    )


def test_classify_known_codes():
    assert classify("retry_exhausted").error_class == ToolErrorClass.POLICY
    assert classify("retry_exhausted").action == ToolErrorAction.PUBLISH_UNAVAILABLE
    assert classify("unknown_tool").action == ToolErrorAction.PUBLISH_UNAVAILABLE
    assert classify("empty_result").error_class == ToolErrorClass.EMPTY
    assert classify("tool_timeout").error_class == ToolErrorClass.TRANSIENT
    assert classify("need_city").error_class == ToolErrorClass.INVALID_INPUT
    assert classify("finalign_unavailable").error_class == ToolErrorClass.COMPENSATE
    assert classify("finalign_unavailable").action == ToolErrorAction.SURFACE
    assert classify("iwencai_not_configured").error_class == ToolErrorClass.UNAVAILABLE
    assert classify("cancelled").error_class == ToolErrorClass.CONTROL


def test_classify_infers_unlisted_codes():
    assert classify("iwencai_gateway_timeout").error_class == ToolErrorClass.TRANSIENT
    assert classify("pdf_not_found").error_class == ToolErrorClass.EMPTY
    assert classify("service_unavailable").error_class == ToolErrorClass.UNAVAILABLE
    assert classify("invalid_year").error_class == ToolErrorClass.INVALID_INPUT


def test_error_result_stamps_class_and_guidance():
    payload = error_result("finalign_unavailable")
    assert payload["ok"] is False
    assert payload["error_class"] == "compensate"
    assert "finalign_analyze" in str(payload["model_guidance"])


def test_enrich_adds_class_to_bare_failure():
    payload = enrich_tool_result({"ok": False, "error": "empty_result"})
    assert payload["error_class"] == "empty"
    assert payload["model_guidance"]


def test_policy_without_success_publishes():
    decision = decide_after_tools(
        [{"ok": False, "error": "unknown_tool"}],
        events=[_event(data={"name": "missing", "ok": False, "error": "unknown_tool"})],
        turn=1,
    )
    assert decision.action == "publish"
    assert decision.text == USER_UNAVAILABLE_HINT


def test_policy_with_other_success_does_not_publish():
    events = [
        _event(data={"name": "lookup_ok", "ok": True, "content": "100"}),
        _event(data={"name": "missing", "ok": False, "error": "unknown_tool"}),
    ]
    decision = decide_after_tools(
        [{"ok": False, "error": "unknown_tool"}],
        events=events,
        turn=1,
    )
    assert decision.action == "continue"


def test_compensate_continues_on_main_path():
    decision = decide_after_tools(
        [{"ok": False, "error": "finalign_unavailable"}],
        events=[
            _event(data={"name": "lookup_ok", "ok": True, "content": "100"}),
            _event(
                data={
                    "name": "finalign_analyze",
                    "ok": False,
                    "error": "finalign_unavailable",
                    "error_class": "compensate",
                }
            ),
        ],
        turn=1,
    )
    assert decision.action == "continue"


def test_empty_failed_twice_injects_give_up():
    events = [
        _event(event_type="tool/call", data={"name": "lookup_fail", "call_id": "c1"}),
        _event(event_type="tool/call", data={"name": "lookup_fail", "call_id": "c2"}),
        _event(
            data={
                "name": "lookup_fail",
                "ok": False,
                "error": "empty_result",
                "error_class": "empty",
            }
        ),
        _event(
            data={
                "name": "lookup_fail",
                "ok": False,
                "error": "empty_result",
                "error_class": "empty",
            }
        ),
    ]
    decision = decide_after_tools(
        [{"ok": False, "error": "empty_result"}],
        events=events,
        turn=1,
    )
    assert decision.action == "inject"
    assert USER_UNAVAILABLE_HINT in decision.text


def test_invalid_input_failed_twice_does_not_inject_give_up():
    events = [
        _event(event_type="tool/call", data={"name": "lookup_fail", "call_id": "c1"}),
        _event(event_type="tool/call", data={"name": "lookup_fail", "call_id": "c2"}),
        _event(
            data={
                "name": "lookup_fail",
                "ok": False,
                "error": "malformed_arguments",
                "error_class": "invalid_input",
            }
        ),
        _event(
            data={
                "name": "lookup_fail",
                "ok": False,
                "error": "malformed_arguments",
                "error_class": "invalid_input",
            }
        ),
    ]
    decision = decide_after_tools(
        [{"ok": False, "error": "malformed_arguments"}],
        events=events,
        turn=1,
    )
    assert decision.action == "continue"
