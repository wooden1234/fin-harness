"""Postgres session 日志。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from sqlalchemy import func, select, text

from harness.session.envelope import event_from_row, normalize_draft
from harness.session.reducer import SessionState, reduce_session, state_from_dict, state_to_dict
from harness.session.store import SessionHeader
from harness.session.types import EventDraft, SessionEvent, new_id, utcnow


class PostgresSessionStore:
    persist_assistant_chunks = False

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

    async def find_by_conversation(
        self, *, tenant_id: str, user_id: str, conversation_id: str | int
    ) -> SessionHeader | None:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import AgentSession

        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(AgentSession).where(
                    AgentSession.conversation_id == str(conversation_id),
                    AgentSession.tenant_id == str(tenant_id),
                    AgentSession.user_id == str(user_id),
                )
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
        from app.models.agent.session import SessionEventLog

        async with AsyncSessionLocal() as db:
            result = await db.scalars(
                select(SessionEventLog)
                .where(
                    SessionEventLog.session_id == session_id,
                    SessionEventLog.seq > after_seq,
                )
                .order_by(SessionEventLog.seq)
            )
            rows = list(result)
            return [await _event_from_log(db, row) for row in rows]

    async def append(self, session_id: str, draft: EventDraft) -> SessionEvent:
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import AgentSession, SessionEventLog

        normalized = normalize_draft(draft)
        async with AsyncSessionLocal() as db:
            seq = await db.scalar(
                text(
                    "UPDATE app.agent_sessions "
                    "SET next_seq = next_seq + 1, last_event_seq = next_seq, updated_at = NOW() "
                    "WHERE session_id = :session_id "
                    "RETURNING next_seq - 1"
                ),
                {"session_id": session_id},
            )
            if seq is None:
                raise KeyError(session_id)
            created_at = utcnow()
            event_id = normalized.event_id or new_id()
            payload_kind, payload_id, metadata = await _write_domain_record(
                db,
                session_id=session_id,
                seq=int(seq),
                event_id=event_id,
                draft=normalized,
                created_at=created_at,
            )
            row = SessionEventLog(
                session_id=session_id,
                seq=int(seq),
                event_id=event_id,
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
                payload_kind=payload_kind,
                payload_id=payload_id,
                metadata_=metadata,
                created_at=created_at,
            )
            db.add(row)
            await _update_session_index(db, AgentSession, session_id, normalized, event_id=event_id)
            await _maybe_write_state_snapshot(
                db, session_id=session_id, seq=int(seq), event=_event_from_draft(
                    seq=int(seq), event_id=event_id, draft=normalized, created_at=created_at
                ),
            )
            await db.commit()
            event = await _event_from_log(db, row)
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

    async def load_state(self, session_id: str) -> SessionState:
        """Restore the latest persisted reducer state plus only its event tail."""
        from app.core.database import AsyncSessionLocal
        from app.models.agent.session import SessionEventLog, SessionStateSnapshot

        async with AsyncSessionLocal() as db:
            snapshot = await db.scalar(
                select(SessionStateSnapshot)
                .where(SessionStateSnapshot.session_id == session_id)
                .order_by(SessionStateSnapshot.snapshot_seq.desc())
                .limit(1)
            )
            state = state_from_dict(snapshot.state) if snapshot is not None else SessionState()
            rows = list(await db.scalars(
                select(SessionEventLog)
                .where(SessionEventLog.session_id == session_id, SessionEventLog.seq > state.last_event_seq)
                .order_by(SessionEventLog.seq)
            ))
            return reduce_session(state, [await _event_from_log(db, row) for row in rows])


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


def _event_from_draft(*, seq: int, event_id: str, draft, created_at) -> SessionEvent:
    return event_from_row(
        seq=seq, event_id=event_id, event_type=draft.event_type,
        schema_version=draft.schema_version, run_id=draft.run_id, turn=draft.turn,
        step=draft.step, causation_seq=draft.causation_seq, correlation_id=draft.correlation_id,
        surface_op=draft.surface_op, source_event_seqs=draft.source_event_seqs,
        visibility=draft.visibility, data=draft.data, created_at=created_at,
    )


_SNAPSHOT_EVENTS = frozenset({
    "approval/asked", "approval/decided", "answer/published", "compaction/summary", "turn/end",
})
_SNAPSHOT_INTERVAL = 50


async def _maybe_write_state_snapshot(db, *, session_id: str, seq: int, event: SessionEvent) -> None:
    """Persist bounded recovery points; replay fills the gap between snapshots."""
    if event.event_type not in _SNAPSHOT_EVENTS and seq % _SNAPSHOT_INTERVAL:
        return
    from app.models.agent.session import SessionEventLog, SessionStateSnapshot

    previous = await db.scalar(
        select(SessionStateSnapshot)
        .where(SessionStateSnapshot.session_id == session_id)
        .order_by(SessionStateSnapshot.snapshot_seq.desc())
        .limit(1)
    )
    base = state_from_dict(previous.state) if previous is not None else SessionState()
    rows = list(await db.scalars(
        select(SessionEventLog)
        .where(SessionEventLog.session_id == session_id, SessionEventLog.seq > base.last_event_seq)
        .order_by(SessionEventLog.seq)
    ))
    prior_events = [await _event_from_log(db, row) for row in rows]
    state = reduce_session(base, [*prior_events, event])
    db.add(SessionStateSnapshot(
        snapshot_id=new_id(), session_id=session_id, snapshot_seq=seq,
        current_run_id=state.current_run_id, current_turn=state.current_turn,
        current_step=state.current_step, run_status=state.status,
        pending_approval_id=state.pending_approval_id,
        pending_call_ids=list(state.pending_tool_calls),
        published_message_id=state.published_event_id,
        latest_compaction_id=state.latest_compaction_id,
        state=state_to_dict(state),
    ))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def _write_domain_record(db, *, session_id: str, seq: int, event_id: str, draft, created_at):
    from app.models.agent.session import (
        SessionApproval,
        SessionCompaction,
        SessionMessage,
        SessionModelCall,
        SessionTodoSnapshot,
        SessionToolAttempt,
        SessionToolCall,
    )

    event_type = draft.event_type
    data = dict(draft.data)
    run_id = str(draft.run_id or "")
    turn = int(draft.turn or 0)
    step = int(draft.step or 0)

    if event_type in {"user/message", "assistant/message", "answer/published"}:
        role = "user" if event_type == "user/message" else "assistant"
        content = str(data.get("content") or data.get("markdown") or "")
        db.add(SessionMessage(
            message_id=event_id, session_id=session_id, event_seq=seq, run_id=draft.run_id,
            turn=draft.turn, step=draft.step, role=role, source=data.get("source"),
            content=content, tool_calls=data.get("tool_calls"), data=data,
            published=event_type == "answer/published", created_at=created_at,
        ))
        if event_type == "assistant/message" and run_id and step:
            model_call = await db.scalar(select(SessionModelCall).where(
                SessionModelCall.session_id == session_id,
                SessionModelCall.run_id == run_id,
                SessionModelCall.step == step,
                SessionModelCall.status == "running",
            ).order_by(SessionModelCall.created_at.desc()))
            if model_call is not None:
                model_call.status = "succeeded"
                model_call.response_message_id = event_id
                model_call.completed_at = created_at
        return "message", event_id, {}

    if event_type == "request/header":
        tools = data.get("tools") or []
        system = str(data.get("system") or "")
        db.add(SessionModelCall(
            model_call_id=event_id, session_id=session_id, run_id=run_id, turn=turn, step=step,
            status="running", system_prompt_hash=_canonical_hash(system), tools_hash=_canonical_hash(tools),
            source_to_seq=seq - 1, request_data=data, created_at=created_at,
        ))
        return "model_call", event_id, {}

    if event_type == "tool/call":
        provider_call_id = str(data.get("call_id") or event_id)
        arguments = data.get("arguments") if isinstance(data.get("arguments"), dict) else {"_raw": data.get("arguments")}
        arguments_hash = _canonical_hash(arguments)
        tool_call_id = event_id
        idempotency_key = _canonical_hash({
            "session_id": session_id, "run_id": run_id, "provider_call_id": provider_call_id,
            "name": data.get("name"), "arguments": arguments,
        })
        db.add(SessionToolCall(
            tool_call_id=tool_call_id, provider_call_id=provider_call_id, session_id=session_id,
            run_id=run_id, turn=turn, step=step, tool_name=str(data.get("name") or "unknown"),
            arguments=arguments, arguments_hash=arguments_hash, idempotency_key=idempotency_key,
            status="pending", attempt_count=0, call_data=data, created_at=created_at,
        ))
        return "tool_call", tool_call_id, {}

    if event_type == "tool/result":
        provider_call_id = str(data.get("call_id") or "")
        call = await db.scalar(select(SessionToolCall).where(
            SessionToolCall.session_id == session_id,
            SessionToolCall.provider_call_id == provider_call_id,
        ).order_by(SessionToolCall.created_at.desc()))
        if call is None:
            arguments_hash = _canonical_hash({})
            call = SessionToolCall(
                tool_call_id=new_id(), provider_call_id=provider_call_id or None, session_id=session_id,
                run_id=run_id, turn=turn, step=step, tool_name=str(data.get("name") or "unknown"),
                arguments={}, arguments_hash=arguments_hash,
                idempotency_key=_canonical_hash({"session_id": session_id, "event_id": event_id}),
                status="pending", attempt_count=0, call_data={}, created_at=created_at,
            )
            db.add(call)
        call.attempt_count = int(call.attempt_count or 0) + 1
        call.status = "succeeded" if bool(data.get("ok", True)) else "failed"
        call.model_content = str(data.get("content") or "")
        call.result_preview = {key: data.get(key) for key in ("ok", "error", "error_class", "evidence_id") if key in data}
        call.result_hash = _canonical_hash(call.model_content)
        call.evidence_id = str(data.get("evidence_id") or "") or None
        call.error_code = str(data.get("error") or "") or None
        call.error_class = str(data.get("error_class") or "") or None
        call.completed_at = created_at
        db.add(SessionToolAttempt(
            attempt_id=event_id, tool_call_id=call.tool_call_id, attempt_no=call.attempt_count,
            status=call.status, error_code=call.error_code, event_data=data,
            model_content=call.model_content, result_hash=call.result_hash,
            started_at=created_at, completed_at=created_at,
        ))
        return "tool_attempt", event_id, {}

    if event_type == "approval/asked":
        approval_id = str(data.get("approval_id") or draft.correlation_id or event_id)
        db.add(SessionApproval(
            approval_id=approval_id, session_id=session_id, run_id=run_id,
            status="pending", requested_data=data, requested_at=created_at,
        ))
        return "approval", approval_id, {}

    if event_type == "approval/decided":
        approval_id = str(data.get("approval_id") or draft.correlation_id or "")
        approval = await db.get(SessionApproval, approval_id) if approval_id else None
        if approval is not None:
            approval.status = "allowed" if str(data.get("decision")) == "allow" else "denied"
            approval.decision = str(data.get("decision") or "")
            approval.decided_at = created_at
        return "approval", approval_id or None, data

    if event_type == "compaction/summary":
        sources = list(draft.source_event_seqs)
        db.add(SessionCompaction(
            compaction_id=event_id, session_id=session_id, run_id=run_id, turn=turn,
            status="committed", source_from_seq=min(sources) if sources else 0,
            source_to_seq=max(sources) if sources else 0, source_event_seqs=sources,
            schema_version=int(data.get("compaction_schema_version") or 1),
            summary=str(data.get("summary") or data.get("content") or ""),
            structured_summary=data.get("structured"), event_data=data,
            summary_tokens=data.get("tokens"), created_at=created_at, committed_at=created_at,
        ))
        return "compaction", event_id, {}

    if event_type == "todo/write":
        version = await db.scalar(select(func.count()).select_from(SessionTodoSnapshot).where(
            SessionTodoSnapshot.session_id == session_id, SessionTodoSnapshot.turn == turn,
        ))
        db.add(SessionTodoSnapshot(
            snapshot_id=event_id, session_id=session_id, event_seq=seq, run_id=run_id,
            turn=turn, version=int(version or 0) + 1, todos=list(data.get("todos") or []),
            event_data=data, created_at=created_at,
        ))
        return "todo", event_id, {}

    return None, None, data


async def _update_session_index(db, model, session_id: str, draft, *, event_id: str) -> None:
    row = await db.get(model, session_id)
    if row is None:
        return
    if draft.turn is not None:
        row.current_turn = int(draft.turn)
    if draft.step is not None:
        row.current_step = int(draft.step)
    if draft.run_id:
        row.current_run_id = draft.run_id
    if draft.event_type == "turn/start":
        row.status = "running"
    elif draft.event_type == "turn/end":
        reason = str(draft.data.get("reason") or "completed")
        row.status = "waiting_approval" if reason == "waiting_approval" else reason
    elif draft.event_type == "approval/asked":
        row.status = "waiting_approval"
        row.pending_approval_id = str(draft.data.get("approval_id") or "") or None
    elif draft.event_type == "approval/decided":
        row.status = "running"
        row.pending_approval_id = None
    elif draft.event_type == "compaction/summary":
        row.latest_compaction_id = event_id
        row.last_compacted_seq = max(draft.source_event_seqs, default=0)


async def _event_from_log(db, row: Any) -> SessionEvent:
    from app.models.agent.session import (
        SessionApproval, SessionCompaction, SessionMessage, SessionModelCall,
        SessionTodoSnapshot, SessionToolAttempt, SessionToolCall,
    )
    data = dict(row.metadata_ or {})
    if row.payload_kind == "message" and row.payload_id:
        item = await db.get(SessionMessage, row.payload_id)
        if item is not None:
            data = dict(item.data or {})
    elif row.payload_kind == "model_call" and row.payload_id:
        item = await db.get(SessionModelCall, row.payload_id)
        if item is not None:
            data = dict(item.request_data or {})
    elif row.payload_kind == "tool_call" and row.payload_id:
        item = await db.get(SessionToolCall, row.payload_id)
        if item is not None:
            data = dict(item.call_data or {})
    elif row.payload_kind == "tool_attempt" and row.payload_id:
        item = await db.get(SessionToolAttempt, row.payload_id)
        if item is not None:
            data = dict(item.event_data or {})
    elif row.payload_kind == "approval" and row.payload_id:
        item = await db.get(SessionApproval, row.payload_id)
        if item is not None and row.event_type == "approval/asked":
            data = dict(item.requested_data or {})
    elif row.payload_kind == "compaction" and row.payload_id:
        item = await db.get(SessionCompaction, row.payload_id)
        if item is not None:
            data = dict(item.event_data or {})
    elif row.payload_kind == "todo" and row.payload_id:
        item = await db.get(SessionTodoSnapshot, row.payload_id)
        if item is not None:
            data = dict(item.event_data or {})
    return event_from_row(
        seq=int(row.seq), event_id=row.event_id, event_type=row.event_type,
        schema_version=int(row.schema_version), run_id=row.run_id, turn=row.turn, step=row.step,
        causation_seq=row.causation_seq, correlation_id=row.correlation_id,
        surface_op=row.surface_op, source_event_seqs=row.source_event_seqs or (),
        visibility=row.visibility, data=data, created_at=row.created_at,
    )
