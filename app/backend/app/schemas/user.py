from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    status: str
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AuthUser(UserResponse):
    """JWT 鉴权主体 DTO；不含 password_hash，不可当作 ORM。"""

    tenant_id: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
