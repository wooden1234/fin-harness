from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True, index=True)
    # 旧数据迁移到 default 租户；新认证主体必须携带真实租户 ID。
    tenant_id = Column(String(36), nullable=False, default="default", index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    status = Column(String(20), default="active")
    role = Column(String(32), nullable=False, default="user", index=True)

    conversations = relationship(
        "Conversation",
        back_populates="user",
        foreign_keys="Conversation.user_id",
        cascade="all, delete-orphan",
    )
