"""同一工具在一轮内最多尝试 2 次（含 1 次重试）。"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping, Sequence

from harness.session.types import SessionEvent
from harness.tools.errors import ToolErrorClass, classify, error_result, publishes_if_no_success

MAX_ATTEMPTS_PER_TOOL = 2
UNLIMITED_TOOLS = frozenset({"todo_write"})
USER_UNAVAILABLE_HINT = (
    "没有找到匹配的数据工具或技能，或查询已失败。请换个问法，或稍后再试。"
)
RETRY_EXHAUSTED_MESSAGE = (
    "同一工具本轮已调用过一次并重试过一次，不能再调用。"
    "若没有更匹配的工具或技能，请立即用下面原话回复用户："
    f"{USER_UNAVAILABLE_HINT}"
)
UNKNOWN_TOOL_MESSAGE = (
    "没有名为该名称的工具或技能。"
    "请立即用下面原话回复用户："
    f"{USER_UNAVAILABLE_HINT}"
)


def is_limited_tool(name: str) -> bool:
    return str(name or "").strip() not in UNLIMITED_TOOLS and bool(str(name or "").strip())


def tool_attempt_counts(events: Sequence[SessionEvent], *, turn: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        if event.turn != turn or event.event_type != "tool/call":
            continue
        name = str(event.data.get("name") or "").strip()
        if not is_limited_tool(name):
            continue
        counts[name] += 1
    return dict(counts)


def allocate_tool_attempts(
    calls: Sequence[Any],
    *,
    prior_counts: Mapping[str, int],
) -> tuple[set[str], dict[str, int]]:
    """返回本批应拦截的 call_id，以及更新后的计数。"""
    counts = dict(prior_counts)
    blocked: set[str] = set()
    for call in calls:
        name = str(getattr(call, "name", "") or "").strip()
        call_id = str(getattr(call, "call_id", "") or "")
        if not is_limited_tool(name):
            continue
        used = counts.get(name, 0)
        if used >= MAX_ATTEMPTS_PER_TOOL:
            blocked.add(call_id)
            continue
        counts[name] = used + 1
    return blocked, counts


def retry_exhausted_result(name: str) -> dict[str, Any]:
    return error_result(
        "retry_exhausted",
        name=name,
        message=RETRY_EXHAUSTED_MESSAGE,
        user_hint=USER_UNAVAILABLE_HINT,
    )


def unknown_tool_result(name: str) -> dict[str, Any]:
    return error_result(
        "unknown_tool",
        name=name,
        message=UNKNOWN_TOOL_MESSAGE,
        user_hint=USER_UNAVAILABLE_HINT,
    )


def blocked_retries(
    calls: Sequence[Any], *, events: Sequence[SessionEvent], turn: int
) -> dict[str, dict[str, Any]]:
    """拦截确定无意义的重试；参数类/空结果仅拦截相同参数。"""
    calls_by_id: dict[str, SessionEvent] = {}
    latest_failure: dict[str, SessionEvent] = {}
    for event in events:
        if event.turn != turn:
            continue
        if event.event_type == "tool/call":
            calls_by_id[str(event.data.get("call_id") or "")] = event
        elif event.event_type == "tool/result" and event.data.get("ok") is False:
            name = str(event.data.get("name") or "").strip()
            if name:
                latest_failure[name] = event

    blocked: dict[str, dict[str, Any]] = {}
    for call in calls:
        name = str(getattr(call, "name", "") or "").strip()
        call_id = str(getattr(call, "call_id", "") or "")
        failure = latest_failure.get(name)
        if failure is None:
            continue
        error_class = str(failure.data.get("error_class") or "")
        if not error_class:
            error_class = classify(str(failure.data.get("error") or "")).error_class.value
        if error_class == ToolErrorClass.TRANSIENT.value:
            continue
        if error_class in {ToolErrorClass.INVALID_INPUT.value, ToolErrorClass.EMPTY.value}:
            previous = calls_by_id.get(str(failure.data.get("call_id") or ""))
            if previous is None or _canonical_arguments(previous.data.get("arguments")) != _canonical_arguments(
                getattr(call, "arguments", "")
            ):
                continue
            code = "retry_requires_change"
        elif error_class == ToolErrorClass.UNAVAILABLE.value:
            code = "unavailable_retry_blocked"
        elif error_class == ToolErrorClass.CONTRACT.value:
            code = "contract_retry_blocked"
        else:
            continue
        blocked[call_id] = error_result(code, name=name)
    return blocked


def _canonical_arguments(raw: Any) -> str:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(raw or "").strip()


def turn_has_successful_data_tool(events: Sequence[SessionEvent], *, turn: int) -> bool:
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        name = str(event.data.get("name") or "").strip()
        if not is_limited_tool(name):
            continue
        if event.data.get("ok") is True:
            return True
    return False


def failed_twice_without_success(events: Sequence[SessionEvent], *, turn: int) -> bool:
    attempts = tool_attempt_counts(events, turn=turn)
    succeeded: set[str] = set()
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        name = str(event.data.get("name") or "").strip()
        if is_limited_tool(name) and event.data.get("ok") is True:
            succeeded.add(name)
    return any(
        count >= MAX_ATTEMPTS_PER_TOOL and name not in succeeded
        for name, count in attempts.items()
    )


def should_publish_unavailable(
    results: Sequence[Mapping[str, Any]],
    *,
    events: Sequence[SessionEvent],
    turn: int,
) -> bool:
    """未知工具或重试耗尽等 policy 类错误，且本轮没有成功的数据工具。"""
    items = [item for item in results if isinstance(item, Mapping)]
    if not any(publishes_if_no_success(item) for item in items):
        return False
    return not turn_has_successful_data_tool(events, turn=turn)
