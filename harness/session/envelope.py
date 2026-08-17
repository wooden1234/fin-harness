"""事件信封校验。"""

from __future__ import annotations

from typing import Any, Mapping

from harness.contracts.errors import SessionFormatError
from harness.contracts.events import (
    EVENT_TYPES,
    SURFACE_EVENT_TYPES,
    TURN_END_REASONS,
    USER_MESSAGE_SOURCES,
)
from harness.session.types import EventDraft, SessionEvent

_DEFAULT_VISIBILITY: dict[str, str] = {
    "turn/start": "internal",
    "turn/end": "public",
    "step/start": "internal",
    "step/end": "internal",
    "user/message": "public",
    "assistant/chunk": "internal",
    "assistant/message": "internal",
    "tool/call": "public",
    "tool/result": "public",
    "request/header": "internal",
    "request/context": "internal",
    "todo/write": "public",
    "inbox/spliced": "internal",
    "inbox/claimed": "internal",
    "inbox/discarded": "internal",
    "approval/asked": "public",
    "approval/decided": "public",
    "compaction/start": "internal",
    "compaction/summary": "internal",
    "compaction/end": "internal",
    "answer/published": "public",
    "session/seed-end": "internal",
    "invariant/violation": "internal",
}

_DEFAULT_SURFACE_OP: dict[str, str] = {
    "user/message": "append",
    "assistant/message": "append",
    "tool/result": "append",
    "compaction/summary": "replace",
}


def default_visibility(event_type: str) -> str:
    return _DEFAULT_VISIBILITY.get(event_type, "internal")


def _require_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SessionFormatError(f"{path} 的键必须是字符串")
            _require_json_value(item, path=f"{path}.{key}")
        return
    raise SessionFormatError(f"{path} 不是 JSON 值: {type(value).__name__}")


def validate_event_type(event_type: str, data: Mapping[str, Any]) -> None:
    if event_type in EVENT_TYPES:
        return
    if data.get("ignorable") is True:
        return
    raise SessionFormatError(f"未知事件类型且不可忽略: {event_type}")


def normalize_draft(draft: EventDraft) -> EventDraft:
    validate_event_type(draft.event_type, draft.data)
    _require_json_value(dict(draft.data), path="data")
    visibility = draft.visibility or default_visibility(draft.event_type)
    if visibility not in {"public", "internal"}:
        raise SessionFormatError(f"非法 visibility: {visibility}")
    if draft.event_type == "assistant/chunk" and visibility != "internal":
        raise SessionFormatError("assistant/chunk 必须 visibility=internal")
    surface_op = draft.surface_op
    if surface_op == "none" and draft.event_type in SURFACE_EVENT_TYPES:
        surface_op = _DEFAULT_SURFACE_OP.get(draft.event_type, "append")
    if surface_op not in {"none", "append", "replace"}:
        raise SessionFormatError(f"非法 surface_op: {surface_op}")
    if surface_op != "none" and draft.event_type not in SURFACE_EVENT_TYPES:
        raise SessionFormatError(f"{draft.event_type} 不能带 surface_op={surface_op}")
    if draft.event_type in {"turn/start", "turn/end"} and draft.turn is None:
        raise SessionFormatError(f"{draft.event_type} 需要 turn")
    if draft.event_type == "turn/end":
        reason = draft.data.get("reason")
        if reason not in TURN_END_REASONS:
            raise SessionFormatError(f"非法 turn/end reason: {reason}")
    if draft.event_type == "user/message":
        source = draft.data.get("source")
        if source not in USER_MESSAGE_SOURCES:
            raise SessionFormatError(f"非法 user/message source: {source}")
        if not str(draft.data.get("content") or "").strip() and source != "skill":
            raise SessionFormatError("user/message 需要 content")
    if draft.event_type == "tool/call":
        if not draft.data.get("call_id") or not draft.data.get("name"):
            raise SessionFormatError("tool/call 需要 call_id 和 name")
    if draft.event_type == "tool/result" and not draft.data.get("call_id"):
        raise SessionFormatError("tool/result 需要 call_id")
    if surface_op == "replace" and not draft.source_event_seqs:
        raise SessionFormatError("surface_op=replace 需要 source_event_seqs")
    return EventDraft(
        event_type=draft.event_type,
        data=dict(draft.data),
        event_id=draft.event_id,
        schema_version=draft.schema_version,
        run_id=draft.run_id,
        turn=draft.turn,
        step=draft.step,
        causation_seq=draft.causation_seq,
        correlation_id=draft.correlation_id,
        surface_op=surface_op,
        source_event_seqs=tuple(draft.source_event_seqs),
        visibility=visibility,
    )


def event_from_row(
    *,
    seq: int,
    event_id: str,
    event_type: str,
    schema_version: int,
    run_id: str | None,
    turn: int | None,
    step: int | None,
    causation_seq: int | None,
    correlation_id: str | None,
    surface_op: str,
    source_event_seqs: list[int] | tuple[int, ...] | None,
    visibility: str,
    data: Mapping[str, Any],
    created_at,
) -> SessionEvent:
    return SessionEvent(
        seq=seq,
        event_id=event_id,
        event_type=event_type,
        schema_version=schema_version,
        run_id=run_id,
        turn=turn,
        step=step,
        causation_seq=causation_seq,
        correlation_id=correlation_id,
        surface_op=surface_op,
        source_event_seqs=tuple(source_event_seqs or ()),
        visibility=visibility,
        data=dict(data or {}),
        created_at=created_at,
    )
