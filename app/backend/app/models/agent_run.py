"""Agent 运行记录模型。"""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, JSON, String, Text, func

from app.core.database import Base


class AgentRunStatus(str, enum.Enum):
    """Agent 运行状态。"""

    ACCEPTED = "accepted"
    RUNNING = "running"
    GRAPH_COMPLETED = "graph_completed"
    PERSISTED = "persisted"
    PERSIST_PENDING = "persist_pending"
    FAILED = "failed"


class AgentRun(Base):
    """记录一次 Agent 执行及其业务持久化结果。"""

    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_user_created", "user_id", "created_at"),
        Index("ix_agent_runs_conversation_created", "conversation_id", "created_at"),
        Index("ix_agent_runs_status_updated", "status", "updated_at"),
        {"schema": "app"},
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(
        ForeignKey("app.users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    conversation_id = Column(
        ForeignKey("app.conversations.id", ondelete="SET NULL"), nullable=True
    )
    thread_id = Column(String(255), nullable=False)
    trace_id = Column(String(100), nullable=True)
    checkpoint_id = Column(String(255), nullable=True)
    summary_snapshot = Column(JSON, nullable=True)
    status = Column(String(32), nullable=False, default=AgentRunStatus.ACCEPTED.value)
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
