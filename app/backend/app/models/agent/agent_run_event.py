"""Agent 运行过程中的结构化上下文与研究事件。"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    func,
)

from app.core.database import Base


class AgentRunEvent(Base):
    __tablename__ = "agent_run_events"
    __table_args__ = (
        Index("ix_agent_run_events_run_id", "run_id", "id"),
        Index("ix_agent_run_events_space", "space_type", "space_id", "id"),
        {"schema": "app"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_key = Column(String(160), nullable=False, unique=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("app.users.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(
        Integer,
        ForeignKey("app.conversations.id", ondelete="CASCADE"),
        nullable=True,
    )
    run_id = Column(
        String(36),
        ForeignKey("app.agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id = Column(String(128), nullable=True)
    agent_id = Column(String(128), nullable=True)
    space_type = Column(String(32), nullable=False)
    space_id = Column(String(255), nullable=False)
    parent_space_id = Column(String(255), nullable=True)
    event_type = Column(String(80), nullable=False)
    estimated_tokens = Column(Integer, nullable=True)
    actual_input_tokens = Column(Integer, nullable=True)
    effective_limit = Column(Integer, nullable=True)
    tokens_before = Column(Integer, nullable=True)
    tokens_after = Column(Integer, nullable=True)
    compaction_round_count = Column(Integer, nullable=False, default=0)
    summary_attempt_count = Column(Integer, nullable=False, default=0)
    snip_count = Column(Integer, nullable=False, default=0)
    provider_retry_count = Column(Integer, nullable=False, default=0)
    details = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
