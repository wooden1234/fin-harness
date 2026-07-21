"""可靠投递用的 Outbox 事件模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, Integer, JSON, String, Text, func

from app.core.database import Base


class OutboxEvent(Base):
    """保存需要异步补偿处理的业务事件。"""

    __tablename__ = "outbox_events"
    __table_args__ = (
        Index("ix_outbox_events_status_available", "status", "available_at"),
        Index("ix_outbox_events_locked_at", "locked_at"),
        {"schema": "app"},
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_key = Column(String(255), nullable=False, unique=True)
    event_type = Column(String(100), nullable=False)
    aggregate_id = Column(String(100), nullable=False)
    payload = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    available_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    locked_at = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

