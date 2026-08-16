"""Session 权威日志表。"""

from __future__ import annotations

from sqlalchemy import (
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
        UniqueConstraint("conversation_id", name="uq_agent_sessions_conversation_id"),
        Index("ix_agent_sessions_user", "tenant_id", "user_id"),
        {"schema": "app"},
    )

    session_id = Column(String(36), primary_key=True)
    conversation_id = Column(String(64), nullable=True)
    tenant_id = Column(String(36), nullable=False, default="default")
    user_id = Column(String(36), nullable=False)
    next_seq = Column(Integer, nullable=False, default=1)
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
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
