"""长期记忆权威记录。"""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String, Text, func

from app.core.database import Base


class MemoryRecord(Base):
    __tablename__ = "memory_records"
    __table_args__ = (
        Index("ix_memory_records_scope_status", "tenant_id", "user_id", "memory_type", "status"),
        Index("ix_memory_records_expiry", "status", "expires_at"),
        {"schema": "app"},
    )

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False, index=True)
    user_id = Column(ForeignKey("app.users.id", ondelete="CASCADE"), nullable=False, index=True)
    memory_type = Column(String(32), nullable=False)
    memory_key = Column(String(64), nullable=False)
    value_json = Column(JSON, nullable=False)
    display_text = Column(Text, nullable=False)
    provenance_json = Column(JSON, nullable=False)
    confidence = Column(Numeric(4, 3), nullable=False, default=1.0)
    consent_status = Column(String(16), nullable=False, default="granted")
    consented_at = Column(DateTime(timezone=True), nullable=True)
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(16), nullable=False, default="active")
    version = Column(Integer, nullable=False, default=1)
    supersedes_id = Column(String(36), nullable=True)
    source_conversation_id = Column(Integer, nullable=True)
    source_message_id = Column(Integer, nullable=True)
    source_run_id = Column(String(36), nullable=True)
    valid_from = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_recalled_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)
