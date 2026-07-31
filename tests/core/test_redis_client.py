from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core import redis_client as redis_client_module
from app.core.redis_client import RedisClient
from app.core.redis_keys import RedisKeyBuilder
from app.core.redis_metrics import reset_metrics, snapshot


class FakeRedis:
    def __init__(self, *, ping_failures: int = 0) -> None:
        self.ping_failures = ping_failures
        self.ping_calls = 0
        self.closed = False
        self.values: dict[str, str] = {}

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self.ping_calls <= self.ping_failures:
            raise RedisConnectionError("unavailable")
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self.values.get(key) for key in keys]

    async def set(
        self,
        key: str,
        value: str,
        **kwargs: Any,
    ) -> bool:
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def eval(
        self,
        _script: str,
        _number_of_keys: int,
        key: str,
        token: str,
    ) -> int:
        if self.values.get(key) != token:
            return 0
        del self.values[key]
        return 1

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_redis_client_retries_transient_connection_error():
    reset_metrics()
    raw = FakeRedis(ping_failures=1)
    client = RedisClient(
        raw,  # type: ignore[arg-type]
        retry_attempts=2,
        retry_base_delay_ms=0,
    )

    assert await client.ping() is True
    assert raw.ping_calls == 2
    assert snapshot()["counters"]["redis_retries_total"] == 1
    assert client.status()["status"] == "ok"


@pytest.mark.asyncio
async def test_redis_client_requires_builder_key_and_positive_ttl():
    raw = FakeRedis()
    client = RedisClient(
        raw,  # type: ignore[arg-type]
        retry_attempts=0,
        retry_base_delay_ms=0,
    )
    key = RedisKeyBuilder(prefix="fin", environment="test").build("cache", "item")

    assert await client.set(key, "value", ttl_seconds=60) is True
    assert await client.get(key) == "value"
    with pytest.raises(TypeError):
        await client.get("manual:key")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await client.set(key, "value", ttl_seconds=0)


@pytest.mark.asyncio
async def test_redis_client_mget_and_token_lock():
    raw = FakeRedis()
    client = RedisClient(
        raw,  # type: ignore[arg-type]
        retry_attempts=0,
        retry_base_delay_ms=0,
    )
    builder = RedisKeyBuilder(prefix="fin", environment="test")
    first = builder.build("cache", "first")
    second = builder.build("cache", "second")
    lock = builder.build("lock", "user")

    await client.set(first, "one", ttl_seconds=60)
    assert await client.mget((first, second)) == ["one", None]
    assert await client.acquire_lock(lock, "owner-1", ttl_seconds=5) is True
    assert await client.acquire_lock(lock, "owner-2", ttl_seconds=5) is False
    assert await client.release_lock(lock, "owner-2") is False
    assert await client.get(lock) == "owner-1"
    assert await client.release_lock(lock, "owner-1") is True
    assert await client.get(lock) is None


@pytest.mark.asyncio
async def test_optional_redis_failure_keeps_client_for_future_recovery(monkeypatch):
    await redis_client_module.close_redis()
    raw = FakeRedis(ping_failures=10)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_ENABLED", True)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_REQUIRED", False)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_RETRY_ATTEMPTS", 0)
    monkeypatch.setattr(
        redis_client_module.redis_async.Redis,
        "from_url",
        lambda *args, **kwargs: raw,
    )

    client = await redis_client_module.init_redis()

    assert client is not None
    assert client.status()["status"] == "degraded"
    assert redis_client_module.get_redis_client() is client
    health = await redis_client_module.redis_health()
    assert health["status"] == "degraded"
    assert health["required"] is False
    await redis_client_module.close_redis()
    assert raw.closed is True


@pytest.mark.asyncio
async def test_required_redis_failure_blocks_initialization(monkeypatch):
    await redis_client_module.close_redis()
    raw = FakeRedis(ping_failures=10)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_ENABLED", True)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_REQUIRED", True)
    monkeypatch.setattr(redis_client_module.settings, "REDIS_RETRY_ATTEMPTS", 0)
    monkeypatch.setattr(
        redis_client_module.redis_async.Redis,
        "from_url",
        lambda *args, **kwargs: raw,
    )

    with pytest.raises(RuntimeError, match="强依赖 Redis 初始化失败"):
        await redis_client_module.init_redis()

    assert redis_client_module.get_redis_client() is None
    assert raw.closed is True
