"""按错误分类决定本步之后是继续、注入提示，还是直接回复用户。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from harness.session.types import SessionEvent
from harness.tools.errors import (
    ToolErrorClass,
    classify,
    enrich_tool_result,
    injects_always,
    publishes_if_no_success,
)
from harness.tools.retry_policy import (
    USER_UNAVAILABLE_HINT,
    failed_twice_without_success,
    turn_has_successful_data_tool,
)

_RETRY_FAMILY = frozenset(
    {
        ToolErrorClass.TRANSIENT.value,
        ToolErrorClass.EMPTY.value,
        ToolErrorClass.UNAVAILABLE.value,
    }
)

GIVE_UP_INJECT = (
    "同一工具已重试一次仍失败。若没有更匹配的工具或技能，请直接用原话回复用户："
    f"{USER_UNAVAILABLE_HINT}"
)


@dataclass(frozen=True, slots=True)
class ToolErrorDecision:
    action: str  # continue | inject | publish
    text: str = ""
    error_class: str = ""


def _error_class_of(result: Mapping[str, Any]) -> str:
    raw = result.get("error_class")
    if raw:
        return str(raw)
    return classify(str(result.get("error") or "")).error_class.value


def _event_error_class(event: SessionEvent) -> str:
    data = event.data or {}
    raw = data.get("error_class")
    if raw:
        return str(raw)
    code = data.get("error")
    if code:
        return classify(str(code)).error_class.value
    content = str(data.get("content") or "")
    if '"error"' in content:
        for key in ("retry_exhausted", "unknown_tool", "empty_result", "finalign_unavailable"):
            if key in content:
                return classify(key).error_class.value
    return ""


def _repeated_failure_is_retry_family(events: Sequence[SessionEvent], *, turn: int) -> bool:
    """两次仍失败时，只对可重试/空结果/源不可用注入放弃提示；参数错误不套这句。"""
    classes: list[str] = []
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        if event.data.get("ok") is True:
            continue
        error_class = _event_error_class(event)
        if error_class:
            classes.append(error_class)
    if not classes:
        return True
    return any(item in _RETRY_FAMILY for item in classes)


def decide_after_tools(
    results: Sequence[Any],
    *,
    events: Sequence[SessionEvent],
    turn: int,
) -> ToolErrorDecision:
    items = [enrich_tool_result(item) for item in results if isinstance(item, Mapping)]
    failures = [item for item in items if item.get("ok") is False]
    no_success = not turn_has_successful_data_tool(events, turn=turn)

    if no_success and any(publishes_if_no_success(item) for item in failures):
        return ToolErrorDecision("publish", USER_UNAVAILABLE_HINT, ToolErrorClass.POLICY.value)

    for item in failures:
        if injects_always(item):
            return ToolErrorDecision(
                "inject",
                str(item.get("model_guidance") or ""),
                _error_class_of(item),
            )

    if failed_twice_without_success(events, turn=turn) and _repeated_failure_is_retry_family(
        events, turn=turn
    ):
        return ToolErrorDecision("inject", GIVE_UP_INJECT, ToolErrorClass.EMPTY.value)

    return ToolErrorDecision("continue")
