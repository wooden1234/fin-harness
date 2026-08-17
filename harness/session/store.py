"""Session 存储：内存实现供测试；Postgres 供产品。"""

from __future__ import annotations

import asyncio
from typing import Protocol, Sequence

from harness.session.envelope import event_from_row, normalize_draft
from harness.session.types import EventDraft, SessionEvent, new_id, utcnow


class SessionHeader:
    def __init__(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
        tenant_id: str = "default",
        user_id: str = "0",
    ) -> None:
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.next_seq = 1
        self.events: list[SessionEvent] = []


class SessionStore(Protocol):
    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionHeader: ...

    async def find_by_conversation(self, conversation_id: str | int) -> SessionHeader | None: ...

    async def get(self, session_id: str) -> SessionHeader: ...

    async def load_events(self, session_id: str, *, after_seq: int = 0) -> list[SessionEvent]: ...

    async def append(self, session_id: str, draft: EventDraft) -> SessionEvent: ...

    async def wait_events(
        self, session_id: str, *, after_seq: int, timeout: float = 0.25
    ) -> list[SessionEvent]: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionHeader] = {}
        self._notifiers: dict[str, asyncio.Event] = {}

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionHeader:
        header = SessionHeader(
            session_id or new_id(),
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        self._sessions[header.session_id] = header
        return header

    async def find_by_conversation(self, conversation_id: str | int) -> SessionHeader | None:
        key = str(conversation_id)
        for header in self._sessions.values():
            if header.conversation_id == key:
                return header
        return None

    async def get(self, session_id: str) -> SessionHeader:
        return self._sessions[session_id]

    async def load_events(self, session_id: str, *, after_seq: int = 0) -> list[SessionEvent]:
        return [event for event in self._sessions[session_id].events if event.seq > after_seq]

    async def append(self, session_id: str, draft: EventDraft) -> SessionEvent:
        header = self._sessions[session_id]
        normalized = normalize_draft(draft)
        seq = header.next_seq
        header.next_seq += 1
        event = event_from_row(
            seq=seq,
            event_id=normalized.event_id or new_id(),
            event_type=normalized.event_type,
            schema_version=normalized.schema_version,
            run_id=normalized.run_id,
            turn=normalized.turn,
            step=normalized.step,
            causation_seq=normalized.causation_seq,
            correlation_id=normalized.correlation_id,
            surface_op=normalized.surface_op,
            source_event_seqs=normalized.source_event_seqs,
            visibility=normalized.visibility,
            data=normalized.data,
            created_at=utcnow(),
        )
        header.events.append(event)
        self._notify(session_id)
        return event

    def _notify(self, session_id: str) -> None:
        current = self._notifiers.get(session_id)
        if current is not None:
            current.set()
        self._notifiers[session_id] = asyncio.Event()

    async def wait_events(
        self, session_id: str, *, after_seq: int, timeout: float = 0.25
    ) -> list[SessionEvent]:
        events = await self.load_events(session_id, after_seq=after_seq)
        if events:
            return events
        waiter = self._notifiers.setdefault(session_id, asyncio.Event())
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except TimeoutError:
            pass
        return await self.load_events(session_id, after_seq=after_seq)


def assert_contiguous(events: Sequence[SessionEvent]) -> None:
    for index, event in enumerate(events, start=1):
        if event.seq != index:
            raise AssertionError(f"seq 不连续: 期望 {index} 得到 {event.seq}")
