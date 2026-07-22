"""长期记忆审计与异步索引事件。"""

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text, func

from app.core.database import Base


class MemoryEvent(Base):
    __tablename__ = "memory_events"
    __table_args__ = ({"schema": "app"},)

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    user_id = Column(ForeignKey("app.users.id", ondelete="CASCADE"), nullable=False, index=True)
    memory_id = Column(ForeignKey("app.memory_records.id", ondelete="SET NULL"), nullable=True)
    event_type = Column(String(32), nullable=False)
    event_key = Column(String(128), nullable=False, unique=True)
    payload_json = Column(JSON, nullable=False)
    actor_type = Column(String(16), nullable=False, default="user")
    actor_id = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
