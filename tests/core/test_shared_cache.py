"""共享缓存协议与 AuthUser 鉴权缓存测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.core.cache import (
    CacheResult,
    cache_get,
    cache_metrics_snapshot,
    cache_set,
    decode_cache_result,
    encode_cache_result,
    make_missing,
    make_record,
    normalize_cache_text,
    reset_cache_metrics,
)
from app.core.redis_keys import redis_keys
from app.schemas.user import AuthUser, UserResponse
from app.services.identity.user_service import (
    auth_user_cache_key,
    normalize_auth_email,
)


def test_normalize_cache_text_collapses_whitespace() -> None:
    assert normalize_cache_text("  a\t b\n c  ") == "a b c"


def test_cache_envelope_roundtrip_and_missing_kind() -> None:
    record = make_record(data_type="auth_user", payload={"id": 1})
    raw = encode_cache_result(record)
    decoded = decode_cache_result(raw, expected_data_type="auth_user")
    assert decoded.kind == "record"
    assert decoded.payload == {"id": 1}

    missing = make_missing(data_type="auth_user")
    missing_decoded = decode_cache_result(
        encode_cache_result(missing),
        expected_data_type="auth_user",
    )
    assert missing_decoded.kind == "missing"
    assert missing_decoded.payload == {}


def test_cache_envelope_rejects_wrong_data_type() -> None:
    raw = encode_cache_result(make_record(data_type="auth_user", payload={}))
    with pytest.raises(ValueError, match="cache_data_type_invalid"):
        decode_cache_result(raw, expected_data_type="query_embedding")


def test_auth_user_compatible_with_user_response() -> None:
    auth_user = AuthUser(
        id=1,
        username="alice",
        email="alice@example.com",
        status="active",
        created_at=datetime.now(timezone.utc),
        last_login=None,
        tenant_id="default",
        role="user",
    )
    response = UserResponse.model_validate(auth_user.model_dump())
    assert response.id == 1
    assert response.email == "alice@example.com"
    assert "tenant_id" not in response.model_dump()


def test_normalize_auth_email() -> None:
    assert normalize_auth_email("  Foo@Bar.COM ") == "foo@bar.com"
    assert auth_user_cache_key("foo@bar.com").value.endswith(
        redis_keys.digest("foo@bar.com")
    )


@pytest.mark.asyncio
async def test_cache_set_get_disabled_is_noop(monkeypatch) -> None:
    reset_cache_metrics()
    key = redis_keys.build("auth", "user", "email", "test")
    assert (
        await cache_set(
            key,
            make_record(data_type="auth_user", payload={"ok": True}),
            domain="auth_user",
            ttl_seconds=60,
            enabled=False,
        )
        is False
    )
    assert (
        await cache_get(
            key,
            domain="auth_user",
            data_type="auth_user",
            enabled=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_cache_metrics_snapshot_shape() -> None:
    reset_cache_metrics()
    from app.core.cache import record_cache_event

    record_cache_event("auth_user", "hit")
    record_cache_event("auth_user", "miss")
    snapshot = cache_metrics_snapshot()
    assert snapshot["domains"]["auth_user"]["hit"] == 1
    assert snapshot["domains"]["auth_user"]["miss"] == 1
