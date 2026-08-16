"""Postgres session 日志。"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select, text

from harness.session.envelope import event_from_row, normalize_draft
from harness.session.store import SessionHeader
from harness.session.types import EventDraft, SessionEvent, new_id, utcnow


class PostgresSessionStore:
    def __init__(self) -> None:
        self._notifiers: dict[str, asyncio.Event] = {}

    async def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionHeader:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import AgentSession

        header = SessionHeader(
            session_id or new_id(),
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            tenant_id=tenant_id,
            user_id=str(user_id),
        )
        async with AsyncSessionLocal() as db:
            db.add(
                AgentSession(
                    session_id=header.session_id,
                    conversation_id=header.conversation_id,
                    tenant_id=header.tenant_id,
                    user_id=header.user_id,
                    next_seq=1,
                )
            )
            await db.commit()
        return header

    async def find_by_conversation(self, conversation_id: str | int) -> SessionHeader | None:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import AgentSession

        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(AgentSession).where(AgentSession.conversation_id == str(conversation_id))
            )
            if row is None:
                return None
            return _header_from_row(row)

    async def get(self, session_id: str) -> SessionHeader:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import AgentSession

        async with AsyncSessionLocal() as db:
            row = await db.get(AgentSession, session_id)
            if row is None:
                raise KeyError(session_id)
            return _header_from_row(row)

    async def load_events(self, session_id: str, *, after_seq: int = 0) -> list[SessionEvent]:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import SessionEventRow

        async with AsyncSessionLocal() as db:
            result = await db.scalars(
                select(SessionEventRow)
                .where(
                    SessionEventRow.session_id == session_id,
                    SessionEventRow.seq > after_seq,
                )
                .order_by(SessionEventRow.seq)
            )
            return [_event_from_orm(row) for row in result]

    async def append(self, session_id: str, draft: EventDraft) -> SessionEvent:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import SessionEventRow

        normalized = normalize_draft(draft)
        async with AsyncSessionLocal() as db:
            seq = await db.scalar(
                text(
                    "UPDATE app.agent_sessions "
                    "SET next_seq = next_seq + 1, updated_at = NOW() "
                    "WHERE session_id = :session_id "
                    "RETURNING next_seq - 1"
                ),
                {"session_id": session_id},
            )
            if seq is None:
                raise KeyError(session_id)
            created_at = utcnow()
            row = SessionEventRow(
                session_id=session_id,
                seq=int(seq),
                event_id=normalized.event_id or new_id(),
                event_type=normalized.event_type,
                schema_version=normalized.schema_version,
                run_id=normalized.run_id,
                turn=normalized.turn,
                step=normalized.step,
                causation_seq=normalized.causation_seq,
                correlation_id=normalized.correlation_id,
                surface_op=normalized.surface_op,
                source_event_seqs=list(normalized.source_event_seqs),
                visibility=normalized.visibility,
                data=dict(normalized.data),
                created_at=created_at,
            )
            db.add(row)
            await db.commit()
            event = _event_from_orm(row)
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


def _header_from_row(row: Any) -> SessionHeader:
    header = SessionHeader(
        row.session_id,
        conversation_id=row.conversation_id,
        tenant_id=row.tenant_id,
        user_id=str(row.user_id),
    )
    header.next_seq = int(row.next_seq)
    return header


def _event_from_orm(row: Any) -> SessionEvent:
    return event_from_row(
        seq=int(row.seq),
        event_id=row.event_id,
        event_type=row.event_type,
        schema_version=int(row.schema_version),
        run_id=row.run_id,
        turn=row.turn,
        step=row.step,
        causation_seq=row.causation_seq,
        correlation_id=row.correlation_id,
        surface_op=row.surface_op,
        source_event_seqs=row.source_event_seqs or (),
        visibility=row.visibility,
        data=row.data or {},
        created_at=row.created_at,
    )
