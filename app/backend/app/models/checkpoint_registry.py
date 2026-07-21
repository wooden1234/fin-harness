"""Checkpoint 生命周期登记表。"""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, func
from app.core.database import Base


class CheckpointRegistry(Base):
    __tablename__ = "checkpoint_registry"
    __table_args__ = (
        Index("ix_checkpoint_registry_expires_at", "expires_at"),
        Index("ix_checkpoint_registry_conversation", "conversation_id"),
        {"schema": "app"},
    )

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("app.conversations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("app.users.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(255), nullable=False)
    checkpoint_id = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="active")
    expires_at = Column(DateTime(timezone=True), nullable=False)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

