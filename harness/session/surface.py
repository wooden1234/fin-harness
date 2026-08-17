"""从权威日志重建模型可见消息。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from harness.contracts.events import (
    EVENT_TYPES,
    MESSAGE_PROJECT_USER_SOURCES,
    PUBLIC_SSE_EVENT_TYPES,
)
from harness.session.types import SessionEvent


@dataclass(frozen=True, slots=True)
class SurfaceMessage:
    seq: int
    role: str
    content: str
    source: str = "user"
    call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectedChatMessage:
    sender: str
    content: str
    source_seq: int
    source: str


def messages_for_llm(events: Sequence[SessionEvent]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in derive_messages(events):
        if item.role == "user":
            messages.append({"role": "user", "content": item.content, "source": item.source})
        elif item.role == "assistant":
            payload: dict[str, Any] = {"role": "assistant", "content": item.content}
            if item.tool_calls:
                payload["tool_calls"] = list(item.tool_calls)
            messages.append(payload)
        elif item.role == "tool":
            messages.append(
                {"role": "tool", "content": item.content, "call_id": item.call_id}
            )
    return messages


def _committed_replace_seqs(events: Sequence[SessionEvent]) -> set[int]:
    committed: set[int] = set()
    open_start: int | None = None
    for event in events:
        if event.event_type == "compaction/start":
            open_start = event.seq
        elif event.event_type == "compaction/end" and open_start is not None:
            for item in events:
                if item.event_type == "compaction/summary" and open_start < item.seq < event.seq:
                    committed.add(item.seq)
            open_start = None
    return committed


def derive_messages(events: Sequence[SessionEvent]) -> list[SurfaceMessage]:
    committed_replaces = _committed_replace_seqs(events)
    surface: list[SurfaceMessage] = []
    for event in events:
        if event.event_type not in EVENT_TYPES and event.data.get("ignorable") is True:
            continue
        if event.event_type == "compaction/summary":
            if event.seq not in committed_replaces:
                continue
            drop = set(event.source_event_seqs)
            insert_at = next(
                (index for index, item in enumerate(surface) if item.seq in drop),
                len(surface),
            )
            kept: list[SurfaceMessage] = []
            placed = False
            for index, item in enumerate(surface):
                if index == insert_at and not placed:
                    kept.append(
                        SurfaceMessage(
                            seq=event.seq,
                            role="user",
                            content=str(
                                event.data.get("content") or event.data.get("summary") or ""
                            ),
                            source="compaction",
                        )
                    )
                    placed = True
                if item.seq in drop:
                    continue
                kept.append(item)
            if not placed:
                kept.append(
                    SurfaceMessage(
                        seq=event.seq,
                        role="user",
                        content=str(
                            event.data.get("content") or event.data.get("summary") or ""
                        ),
                        source="compaction",
                    )
                )
            surface[:] = kept
            continue
        if event.event_type == "tool/result" and event.surface_op == "replace":
            drop = set(event.source_event_seqs)
            replacement = SurfaceMessage(
                seq=event.seq,
                role="tool",
                content=str(event.data.get("content") or event.data),
                call_id=str(event.data.get("call_id") or "") or None,
                name=str(event.data.get("name") or "") or None,
            )
            replaced = False
            next_surface: list[SurfaceMessage] = []
            for item in surface:
                if item.seq in drop:
                    if not replaced:
                        next_surface.append(replacement)
                        replaced = True
                    continue
                next_surface.append(item)
            if replaced:
                surface[:] = next_surface
            continue
        if event.surface_op != "append":
            continue
        if event.event_type == "user/message":
            message = SurfaceMessage(
                seq=event.seq,
                role="user",
                content=str(event.data.get("content") or ""),
                source=str(event.data.get("source") or "user"),
            )
        elif event.event_type == "assistant/message":
            tool_calls = event.data.get("tool_calls") or ()
            if not isinstance(tool_calls, (list, tuple)):
                tool_calls = ()
            message = SurfaceMessage(
                seq=event.seq,
                role="assistant",
                content=str(event.data.get("content") or ""),
                tool_calls=tuple(tool_calls),
            )
        elif event.event_type == "tool/result":
            payload = {
                key: event.data[key]
                for key in ("ok", "error", "evidence_id", "content")
                if key in event.data
            }
            message = SurfaceMessage(
                seq=event.seq,
                role="tool",
                content=str(payload.get("content") or payload),
                call_id=str(event.data.get("call_id") or "") or None,
                name=str(event.data.get("name") or "") or None,
            )
        else:
            continue
        surface.append(message)
    return list(surface)


def project_chat_messages(events: Sequence[SessionEvent]) -> list[ProjectedChatMessage]:
    rows: list[ProjectedChatMessage] = []
    for event in events:
        if event.event_type == "user/message":
            source = str(event.data.get("source") or "")
            if source not in MESSAGE_PROJECT_USER_SOURCES:
                continue
            rows.append(
                ProjectedChatMessage(
                    sender="user",
                    content=str(event.data.get("content") or ""),
                    source_seq=event.seq,
                    source=source,
                )
            )
        elif event.event_type == "answer/published":
            rows.append(
                ProjectedChatMessage(
                    sender="assistant",
                    content=str(event.data.get("markdown") or event.data.get("content") or ""),
                    source_seq=event.seq,
                    source=str(event.data.get("source") or "published"),
                )
            )
    return rows


def public_sse_events(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    return [
        event
        for event in events
        if event.visibility == "public" and event.event_type in PUBLIC_SSE_EVENT_TYPES
    ]


def product_sse_events(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    failed_turns = {
        event.turn
        for event in events
        if event.event_type == "turn/end" and event.data.get("reason") == "persist_failed"
    }
    return [
        event
        for event in public_sse_events(events)
        if not (event.event_type == "answer/published" and event.turn in failed_turns)
    ]


def project_inbox(events: Sequence[SessionEvent]) -> list[dict[str, Any]]:
    pending: dict[int, dict[str, Any]] = {}
    claimed: set[int] = set()
    discarded: set[int] = set()
    for event in events:
        if event.event_type == "inbox/spliced":
            pending[event.seq] = {
                "seq": event.seq,
                "status": "spliced",
                "correlation_id": event.correlation_id,
                "data": dict(event.data),
            }
        elif event.event_type == "inbox/claimed":
            claimed.add(int(event.data.get("inbox_seq") or 0))
        elif event.event_type == "inbox/discarded":
            discarded.add(int(event.data.get("inbox_seq") or 0))
    return [
        item
        for seq, item in pending.items()
        if seq not in claimed and seq not in discarded
    ]
