"""用户数据导出与删除审计。"""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, func
from app.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("app.users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(64), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    error_message = Column(Text, nullable=True)

