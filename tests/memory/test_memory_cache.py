from types import SimpleNamespace
from typing import Any

import pytest

from app.services.memory import memory_cache, memory_loader
from app.services.memory.memory_cache import CachedMemoryEntry, MemoryCache
from app.services.memory.memory_catalog import preference_definitions


class FakeMemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def mget(self, keys) -> list[str | None]:
        return [self.values.get(key.value) for key in keys]

    async def set(
        self,
        key,
        value: str,
        *,
        ttl_seconds: int,
        only_if_absent: bool = False,
    ) -> bool:
        assert ttl_seconds > 0
        if only_if_absent and key.value in self.values:
            return False
        self.values[key.value] = value
        return True

    async def acquire_lock(
        self,
        key,
        token: str,
        *,
        ttl_seconds: int,
    ) -> bool:
        assert ttl_seconds > 0
        if key.value in self.values:
            return False
        self.values[key.value] = token
        return True

    async def release_lock(self, key, token: str) -> bool:
        if self.values.get(key.value) != token:
            return False
        del self.values[key.value]
        return True

    async def delete(self, key) -> int:
        return int(self.values.pop(key.value, None) is not None)


class BrokenMemoryRedis(FakeMemoryRedis):
    async def mget(self, keys) -> list[str | None]:
        del keys
        raise RuntimeError("redis_down")

    async def set(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        raise RuntimeError("redis_down")


def _record(value: str = "zh-CN"):
    return SimpleNamespace(
        memory_key="response_language",
        memory_type="preference",
        value_json={"value": value},
        version=2,
        expires_at=None,
    )


@pytest.fixture
def deterministic_cache(monkeypatch):
    monkeypatch.setattr(memory_cache.settings, "MEMORY_CACHE_ENABLED", True)
    monkeypatch.setattr(
        memory_cache.settings,
        "MEMORY_CACHE_TTL_JITTER_RATIO",
        0.0,
    )
    monkeypatch.setattr(memory_cache.settings, "MEMORY_CACHE_LOCK_WAIT_MS", 0)


@pytest.mark.asyncio
async def test_cache_aside_reuses_record_and_negative_cache(
    monkeypatch,
    deterministic_cache,
):
    del deterministic_cache
    redis = FakeMemoryRedis()
    database_calls = 0

    async def fake_list_by_keys(**_kwargs):
        nonlocal database_calls
        database_calls += 1
        return [_record()]

    monkeypatch.setattr(memory_cache, "get_redis_client", lambda: redis)
    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )
    requested = ("response_language", "citation_preference")

    first = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=requested,
    )
    second = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=requested,
    )

    assert first.as_dict() == {"response_language": "zh-CN"}
    assert second.as_dict() == first.as_dict()
    assert database_calls == 1


@pytest.mark.asyncio
async def test_redis_failure_and_cache_bypass_return_authoritative_result(
    monkeypatch,
    deterministic_cache,
):
    del deterministic_cache
    database_calls = 0

    async def fake_list_by_keys(**_kwargs):
        nonlocal database_calls
        database_calls += 1
        return [_record("en-US")]

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )
    monkeypatch.setattr(
        memory_cache,
        "get_redis_client",
        lambda: BrokenMemoryRedis(),
    )
    degraded = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=("response_language",),
    )

    redis = FakeMemoryRedis()
    monkeypatch.setattr(memory_cache, "get_redis_client", lambda: redis)
    await MemoryCache.set_loaded(
        tenant_id="tenant-1",
        user_id=7,
        memory_type="preference",
        requested_keys=("response_language",),
        entries=(
            CachedMemoryEntry(
                memory_key="response_language",
                memory_type="preference",
                value="zh-CN",
                version=1,
                expires_at=None,
            ),
        ),
    )
    bypassed = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=("response_language",),
        bypass_cache=True,
    )

    assert degraded.as_dict() == {"response_language": "en-US"}
    assert bypassed.as_dict() == degraded.as_dict()
    assert database_calls == 2


@pytest.mark.asyncio
async def test_invalidate_user_removes_all_preference_cache_keys(
    monkeypatch,
    deterministic_cache,
):
    del deterministic_cache
    redis = FakeMemoryRedis()
    monkeypatch.setattr(memory_cache, "get_redis_client", lambda: redis)
    memory_keys = tuple(preference_definitions())
    await MemoryCache.set_loaded(
        tenant_id="tenant-1",
        user_id=7,
        memory_type="preference",
        requested_keys=memory_keys,
        entries=(),
    )
    assert len(redis.values) == len(memory_keys)

    await MemoryCache.invalidate_user(
        tenant_id="tenant-1",
        user_id=7,
    )

    assert redis.values == {}
