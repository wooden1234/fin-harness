from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index, func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_sequence", "conversation_id", "sequence_no"),
        Index("ix_messages_run_id", "run_id"),
        Index("ix_messages_client_message_id", "client_message_id"),
        {"schema": "app"},
    )

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer,
        ForeignKey("app.conversations.id", ondelete="CASCADE"),
    )
    sender = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    message_type = Column(String(20), default="text")
    run_id = Column(
        String(36),
        ForeignKey("app.agent_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    sequence_no = Column(Integer, nullable=True)
    client_message_id = Column(String(128), nullable=True)
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")
