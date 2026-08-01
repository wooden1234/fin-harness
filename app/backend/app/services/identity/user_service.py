"""用户服务：注册/登录直查数据库；JWT 鉴权可走 AuthUser 缓存。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import (
    cache_delete,
    cache_get,
    cache_set,
    make_missing,
    make_record,
)
from app.core.config import settings
from app.core.hashing import get_password_hash, verify_password
from app.core.logger import get_logger
from app.core.redis_keys import redis_keys
from app.models.identity.user import User
from app.schemas.user import AuthUser, UserCreate

logger = get_logger(service="user_service")

_AUTH_DATA_TYPE = "auth_user"
_AUTH_DOMAIN = "auth_user"


def normalize_auth_email(email: str) -> str:
    """鉴权邮箱归一化：去空白并转小写。"""
    return str(email or "").strip().lower()


def auth_user_cache_key(norm_email: str):
    return redis_keys.build(
        "auth",
        "user",
        "email",
        redis_keys.digest(norm_email),
    )


async def invalidate_auth_user_cache(email: str) -> None:
    """删除鉴权缓存；失败不抛出。"""
    norm_email = normalize_auth_email(email)
    if not norm_email:
        return
    await cache_delete(
        auth_user_cache_key(norm_email),
        domain=_AUTH_DOMAIN,
        enabled=True,
    )


def _auth_user_from_row(row: object) -> AuthUser:
    return AuthUser(
        id=int(row.id),
        username=str(row.username),
        email=str(row.email),
        status=str(row.status or "active"),
        created_at=row.created_at,
        last_login=row.last_login,
        tenant_id=str(row.tenant_id or ""),
        role=str(row.role or "user"),
    )


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, user_data: UserCreate) -> User:
        norm_email = normalize_auth_email(str(user_data.email))
        query = select(User).where(
            or_(
                func.lower(User.email) == norm_email,
                User.username == user_data.username,
            )
        )
        result = await self.db.execute(query)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            if normalize_auth_email(existing_user.email) == norm_email:
                raise ValueError("该邮箱已经被注册！")
            raise ValueError("用户名已被占用！")

        db_user = User(
            username=user_data.username,
            email=norm_email,
            password_hash=get_password_hash(user_data.password),
        )
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)
        # 失效清单：注册成功后清除可能存在的负缓存
        await invalidate_auth_user_cache(norm_email)
        return db_user

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """登录校验始终直查数据库，禁止读取 Auth 缓存。"""
        norm_email = normalize_auth_email(email)
        query = select(User).where(func.lower(User.email) == norm_email)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            logger.warning("User not found")
            return None

        if not verify_password(password, user.password_hash):
            logger.warning("Invalid password for user")
            return None

        user.last_login = datetime.utcnow()
        await self.db.commit()
        # 失效清单：登录更新 last_login 后删除缓存
        await invalidate_auth_user_cache(norm_email)
        return user

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """通用 ORM 查询；不走 Auth 缓存。"""
        norm_email = normalize_auth_email(email)
        query = select(User).where(func.lower(User.email) == norm_email)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_auth_user_by_email(self, email: str) -> AuthUser | None:
        """仅供 JWT 鉴权：返回 AuthUser DTO，可走 Cache-Aside。"""
        norm_email = normalize_auth_email(email)
        if not norm_email:
            return None

        cache_key = auth_user_cache_key(norm_email)
        enabled = bool(settings.AUTH_USER_CACHE_ENABLED)
        cached = await cache_get(
            cache_key,
            domain=_AUTH_DOMAIN,
            data_type=_AUTH_DATA_TYPE,
            enabled=enabled,
        )
        if cached is not None:
            if cached.kind == "missing":
                return None
            try:
                return AuthUser.model_validate(cached.payload)
            except Exception:
                await cache_delete(cache_key, domain=_AUTH_DOMAIN, enabled=True)

        stmt = select(
            User.id,
            User.username,
            User.email,
            User.status,
            User.created_at,
            User.last_login,
            User.tenant_id,
            User.role,
        ).where(func.lower(User.email) == norm_email)
        result = await self.db.execute(stmt)
        row = result.one_or_none()
        if row is None:
            await cache_set(
                cache_key,
                make_missing(data_type=_AUTH_DATA_TYPE),
                domain=_AUTH_DOMAIN,
                ttl_seconds=settings.AUTH_USER_CACHE_NEGATIVE_TTL_SEC,
                enabled=enabled,
            )
            return None

        auth_user = _auth_user_from_row(row)
        await cache_set(
            cache_key,
            make_record(
                data_type=_AUTH_DATA_TYPE,
                payload=auth_user.model_dump(mode="json"),
            ),
            domain=_AUTH_DOMAIN,
            ttl_seconds=settings.AUTH_USER_CACHE_TTL_SEC,
            enabled=enabled,
        )
        return auth_user
