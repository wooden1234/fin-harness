"""会话级租约锁。"""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, func

from app.core.database import Base


class ConversationLock(Base):
    __tablename__ = "conversation_locks"
    __table_args__ = (
        Index("ix_conversation_locks_expires_at", "expires_at"),
        {"schema": "app"},
    )

    conversation_id = Column(
        Integer,
        ForeignKey("app.conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    lock_token = Column(String(36), nullable=False)
    acquired_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

