"""AuthUser 缓存、鉴权依赖与公开响应兼容测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI, HTTPException
import httpx
import pytest

from app.api import auth as auth_api
from app.core import security
from app.core.cache import make_missing, make_record
from app.schemas.user import AuthUser, UserCreate
from app.services.identity import user_service as user_service_module
from app.services.identity.user_service import UserService


def _auth_user(*, role: str = "user", status: str = "active") -> AuthUser:
    return AuthUser(
        id=1,
        username="alice",
        email="alice@example.com",
        status=status,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_login=None,
        tenant_id="tenant-1",
        role=role,
    )


class _NoDatabase:
    async def execute(self, _statement):
        raise AssertionError("缓存命中时不应查询数据库")


@pytest.mark.asyncio
async def test_get_auth_user_positive_cache_hit(monkeypatch) -> None:
    auth_user = _auth_user()

    async def fake_cache_get(*_args, **_kwargs):
        return make_record(
            data_type="auth_user",
            payload=auth_user.model_dump(mode="json"),
        )

    monkeypatch.setattr(user_service_module.settings, "AUTH_USER_CACHE_ENABLED", True)
    monkeypatch.setattr(user_service_module, "cache_get", fake_cache_get)

    result = await UserService(_NoDatabase()).get_auth_user_by_email(
        " ALICE@EXAMPLE.COM "
    )

    assert result == auth_user


@pytest.mark.asyncio
async def test_get_auth_user_negative_cache_hit(monkeypatch) -> None:
    async def fake_cache_get(*_args, **_kwargs):
        return make_missing(data_type="auth_user")

    monkeypatch.setattr(user_service_module.settings, "AUTH_USER_CACHE_ENABLED", True)
    monkeypatch.setattr(user_service_module, "cache_get", fake_cache_get)

    assert (
        await UserService(_NoDatabase()).get_auth_user_by_email(
            "missing@example.com"
        )
        is None
    )


@pytest.mark.asyncio
async def test_get_auth_user_db_miss_writes_negative_cache(monkeypatch) -> None:
    result_proxy = SimpleNamespace(one_or_none=lambda: None)
    db = SimpleNamespace(execute=AsyncMock(return_value=result_proxy))
    writes: list[object] = []

    async def fake_cache_get(*_args, **_kwargs):
        return None

    async def fake_cache_set(*args, **_kwargs):
        writes.append(args[1])
        return True

    monkeypatch.setattr(user_service_module.settings, "AUTH_USER_CACHE_ENABLED", True)
    monkeypatch.setattr(user_service_module, "cache_get", fake_cache_get)
    monkeypatch.setattr(user_service_module, "cache_set", fake_cache_set)

    assert await UserService(db).get_auth_user_by_email("none@example.com") is None
    assert len(writes) == 1
    assert writes[0].kind == "missing"


@pytest.mark.asyncio
async def test_register_and_login_invalidate_auth_cache(monkeypatch) -> None:
    query_result = SimpleNamespace(scalar_one_or_none=lambda: None)
    db = SimpleNamespace(
        execute=AsyncMock(return_value=query_result),
        add=lambda _user: None,
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )
    invalidated: list[str] = []

    async def fake_invalidate(email: str) -> None:
        invalidated.append(email)

    monkeypatch.setattr(user_service_module, "get_password_hash", lambda _: "hash")
    monkeypatch.setattr(
        user_service_module,
        "invalidate_auth_user_cache",
        fake_invalidate,
    )

    await UserService(db).create_user(
        UserCreate(
            username="alice",
            email="ALICE@example.com",
            password="secret",
        )
    )
    assert invalidated == ["alice@example.com"]

    user = SimpleNamespace(
        email="alice@example.com",
        password_hash="hash",
        last_login=None,
    )
    db.execute = AsyncMock(
        return_value=SimpleNamespace(scalar_one_or_none=lambda: user)
    )
    monkeypatch.setattr(user_service_module, "verify_password", lambda *_: True)

    assert await UserService(db).authenticate_user("ALICE@example.com", "secret") is user
    assert invalidated == ["alice@example.com", "alice@example.com"]
    assert user.last_login is not None


@pytest.mark.asyncio
async def test_get_current_user_rejects_inactive_user(monkeypatch) -> None:
    class FakeUserService:
        def __init__(self, _db):
            pass

        async def get_auth_user_by_email(self, _email: str) -> AuthUser:
            return _auth_user(status="disabled")

    monkeypatch.setattr(security.jwt, "decode", lambda *_args, **_kwargs: {"sub": "a@b.com"})
    monkeypatch.setattr(security, "UserService", FakeUserService)

    with pytest.raises(HTTPException) as exc_info:
        await security.get_current_user(token="token", db=object())
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_compliance_user_role_compatibility() -> None:
    assert (
        await security.require_compliance_user(
            current_user=_auth_user(role="tenant_admin")
        )
    ).role == "tenant_admin"

    with pytest.raises(HTTPException) as exc_info:
        await security.require_compliance_user(current_user=_auth_user(role="user"))
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_users_me_response_model_hides_auth_only_fields() -> None:
    app = FastAPI()
    app.include_router(auth_api.router)

    async def current_user_override() -> AuthUser:
        return _auth_user()

    app.dependency_overrides[security.get_current_user] = current_user_override

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/users/me")

    assert response.status_code == 200
    assert response.json() == {
        "username": "alice",
        "email": "alice@example.com",
        "id": 1,
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "last_login": None,
    }
