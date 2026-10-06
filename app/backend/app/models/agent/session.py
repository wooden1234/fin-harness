"""Session 权威日志表。"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)

from app.core.database import Base


class AgentSession(Base):
    __tablename__ = "agent_sessions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "user_id", "conversation_id", name="uq_agent_sessions_owner_conversation"
        ),
        Index("ix_agent_sessions_user", "tenant_id", "user_id"),
        {"schema": "app"},
    )

    session_id = Column(String(36), primary_key=True)
    conversation_id = Column(String(64), nullable=True)
    tenant_id = Column(String(36), nullable=False, default="default")
    user_id = Column(String(36), nullable=False)
    next_seq = Column(Integer, nullable=False, default=1)
    status = Column(String(32), nullable=False, default="idle")
    current_run_id = Column(String(36), nullable=True)
    current_turn = Column(Integer, nullable=False, default=0)
    current_step = Column(Integer, nullable=False, default=0)
    last_event_seq = Column(Integer, nullable=False, default=0)
    pending_approval_id = Column(String(36), nullable=True)
    latest_compaction_id = Column(String(36), nullable=True)
    last_compacted_seq = Column(Integer, nullable=False, default=0)
    latest_snapshot_seq = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SessionEventRow(Base):
    __tablename__ = "session_events"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_session_events_session_seq"),
        UniqueConstraint("event_id", name="uq_session_events_event_id"),
        Index("ix_session_events_session_seq", "session_id", "seq"),
        {"schema": "app"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    seq = Column(Integer, nullable=False)
    event_id = Column(String(36), nullable=False)
    event_type = Column(String(80), nullable=False)
    schema_version = Column(Integer, nullable=False, default=1)
    run_id = Column(String(36), nullable=True)
    turn = Column(Integer, nullable=True)
    step = Column(Integer, nullable=True)
    causation_seq = Column(Integer, nullable=True)
    correlation_id = Column(String(36), nullable=True)
    surface_op = Column(String(16), nullable=False, default="none")
    source_event_seqs = Column(JSON, nullable=False, default=list)
    visibility = Column(String(16), nullable=False, default="internal")
    data = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AgentSessionLease(Base):
    __tablename__ = "agent_session_leases"
    __table_args__ = ({"schema": "app"},)

    session_id = Column(
        String(36),
        ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    owner_id = Column(String(64), nullable=False)
    token = Column(String(36), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    fencing_token = Column(Integer, nullable=False, default=0)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SessionEventLog(Base):
    __tablename__ = "session_event_log"
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_session_event_log_session_seq"),
        Index("ix_session_event_log_session_seq", "session_id", "seq"),
        Index("ix_session_event_log_run", "run_id", "seq"),
        {"schema": "app"},
    )

    event_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(80), nullable=False)
    schema_version = Column(Integer, nullable=False, default=1)
    run_id = Column(String(36), nullable=True)
    turn = Column(Integer, nullable=True)
    step = Column(Integer, nullable=True)
    causation_seq = Column(Integer, nullable=True)
    correlation_id = Column(String(36), nullable=True)
    surface_op = Column(String(16), nullable=False, default="none")
    source_event_seqs = Column(JSON, nullable=False, default=list)
    visibility = Column(String(16), nullable=False, default="internal")
    payload_kind = Column(String(32), nullable=True)
    payload_id = Column(String(128), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SessionMessage(Base):
    __tablename__ = "session_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "event_seq", name="uq_session_messages_event_seq"),
        Index("ix_session_messages_surface", "session_id", "event_seq"),
        {"schema": "app"},
    )

    message_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    event_seq = Column(Integer, nullable=False)
    run_id = Column(String(36), nullable=True)
    turn = Column(Integer, nullable=True)
    step = Column(Integer, nullable=True)
    role = Column(String(20), nullable=False)
    source = Column(String(32), nullable=True)
    content = Column(String, nullable=False, default="")
    tool_calls = Column(JSON, nullable=True)
    data = Column(JSON, nullable=False, default=dict)
    published = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SessionModelCall(Base):
    __tablename__ = "session_model_calls"
    __table_args__ = (Index("ix_session_model_calls_run", "session_id", "run_id", "step"), {"schema": "app"})

    model_call_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(36), nullable=False)
    turn = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    system_prompt_hash = Column(String(64), nullable=True)
    tools_hash = Column(String(64), nullable=True)
    tools_ref = Column(String(256), nullable=True)
    source_from_seq = Column(Integer, nullable=True)
    source_to_seq = Column(Integer, nullable=True)
    request_data = Column(JSON, nullable=False, default=dict)
    response_message_id = Column(String(36), nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    finish_reason = Column(String(32), nullable=True)
    error_code = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SessionToolCall(Base):
    __tablename__ = "session_tool_calls"
    __table_args__ = (
        UniqueConstraint("session_id", "provider_call_id", name="uq_session_tool_provider_call"),
        UniqueConstraint("idempotency_key", name="uq_session_tool_idempotency"),
        Index("ix_session_tool_calls_status", "session_id", "status"),
        {"schema": "app"},
    )

    tool_call_id = Column(String(36), primary_key=True)
    provider_call_id = Column(String(128), nullable=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(36), nullable=False)
    turn = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    tool_name = Column(String(128), nullable=False)
    arguments = Column(JSON, nullable=False, default=dict)
    arguments_hash = Column(String(64), nullable=False)
    idempotency_key = Column(String(160), nullable=False)
    status = Column(String(32), nullable=False)
    attempt_count = Column(Integer, nullable=False, default=0)
    call_data = Column(JSON, nullable=False, default=dict)
    result_preview = Column(JSON, nullable=True)
    model_content = Column(String, nullable=True)
    result_ref = Column(String(256), nullable=True)
    result_hash = Column(String(64), nullable=True)
    evidence_id = Column(String(128), nullable=True)
    error_code = Column(String(128), nullable=True)
    error_class = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SessionToolAttempt(Base):
    __tablename__ = "session_tool_attempts"
    __table_args__ = (
        UniqueConstraint("tool_call_id", "attempt_no", name="uq_session_tool_attempt_no"),
        {"schema": "app"},
    )
    attempt_id = Column(String(36), primary_key=True)
    tool_call_id = Column(String(36), ForeignKey("app.session_tool_calls.tool_call_id", ondelete="CASCADE"), nullable=False)
    attempt_no = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    error_code = Column(String(128), nullable=True)
    event_data = Column(JSON, nullable=False, default=dict)
    model_content = Column(String, nullable=True)
    result_hash = Column(String(64), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SessionApproval(Base):
    __tablename__ = "session_approvals"
    __table_args__ = (Index("ix_session_approvals_status", "session_id", "status"), {"schema": "app"})
    approval_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(36), nullable=False)
    tool_call_id = Column(String(36), nullable=True)
    status = Column(String(32), nullable=False)
    requested_data = Column(JSON, nullable=False, default=dict)
    decision = Column(String(16), nullable=True)
    decided_by = Column(String(64), nullable=True)
    requested_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    decided_at = Column(DateTime(timezone=True), nullable=True)


class SessionCompaction(Base):
    __tablename__ = "session_compactions"
    __table_args__ = (Index("ix_session_compactions_latest", "session_id", "status", "source_to_seq"), {"schema": "app"})
    compaction_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(36), nullable=False)
    turn = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    source_from_seq = Column(Integer, nullable=False, default=0)
    source_to_seq = Column(Integer, nullable=False, default=0)
    source_event_seqs = Column(JSON, nullable=True)
    schema_version = Column(Integer, nullable=False, default=1)
    summary = Column(String, nullable=False, default="")
    structured_summary = Column(JSON, nullable=True)
    event_data = Column(JSON, nullable=False, default=dict)
    summary_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    committed_at = Column(DateTime(timezone=True), nullable=True)


class SessionTodoSnapshot(Base):
    __tablename__ = "session_todo_snapshots"
    __table_args__ = (
        UniqueConstraint("session_id", "turn", "version", name="uq_session_todo_version"),
        {"schema": "app"},
    )
    snapshot_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    event_seq = Column(Integer, nullable=False)
    run_id = Column(String(36), nullable=False)
    turn = Column(Integer, nullable=False)
    version = Column(Integer, nullable=False)
    todos = Column(JSON, nullable=False)
    event_data = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SessionStateSnapshot(Base):
    __tablename__ = "session_state_snapshots"
    __table_args__ = (
        UniqueConstraint("session_id", "snapshot_seq", name="uq_session_state_snapshot_seq"),
        Index("ix_session_state_snapshots_latest", "session_id", "snapshot_seq"),
        {"schema": "app"},
    )

    snapshot_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("app.agent_sessions.session_id", ondelete="CASCADE"), nullable=False)
    snapshot_seq = Column(Integer, nullable=False)
    schema_version = Column(Integer, nullable=False, default=1)
    current_run_id = Column(String(36), nullable=True)
    current_turn = Column(Integer, nullable=True)
    current_step = Column(Integer, nullable=True)
    run_status = Column(String(32), nullable=True)
    pending_approval_id = Column(String(36), nullable=True)
    pending_call_ids = Column(JSON, nullable=False, default=list)
    published_message_id = Column(String(36), nullable=True)
    latest_compaction_id = Column(String(36), nullable=True)
    state = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
