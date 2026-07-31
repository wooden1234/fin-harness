"""长期记忆 Cache-Aside：逐 Key 缓存、负缓存和防击穿锁。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import random
from typing import Any, Iterable
from uuid import uuid4

from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis_client import get_redis_client
from app.core.redis_keys import RedisKey, redis_keys
from app.services.memory.memory_catalog import MEMORY_KEYS, MEMORY_TYPES

logger = get_logger(service="memory_cache")
_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, init=False)
class CachedMemoryEntry:
    memory_key: str
    memory_type: str
    version: int
    expires_at: datetime | None
    _value: Any = field(repr=False)

    def __init__(
        self,
        *,
        memory_key: str,
        memory_type: str,
        value: Any,
        version: int,
        expires_at: datetime | None,
    ) -> None:
        object.__setattr__(self, "memory_key", memory_key)
        object.__setattr__(self, "memory_type", memory_type)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "_value", value)

    @property
    def value(self) -> Any:
        return self._value


@dataclass(frozen=True, slots=True)
class CacheLookup:
    entries: dict[str, CachedMemoryEntry]
    misses: tuple[str, ...]
    redis_available: bool


def _cache_key(
    tenant_id: str,
    user_id: int,
    memory_type: str,
    memory_key: str,
) -> RedisKey:
    return redis_keys.user(
        "memory-cache",
        tenant_id,
        user_id,
        memory_type,
        memory_key,
    )


def _lock_key(
    tenant_id: str,
    user_id: int,
    memory_type: str,
) -> RedisKey:
    return redis_keys.user(
        "memory-lock",
        tenant_id,
        user_id,
        memory_type,
    )


def _cacheable_keys(memory_type: str) -> tuple[str, ...]:
    memory_type_definition = MEMORY_TYPES.get(memory_type)
    if memory_type_definition is None or not memory_type_definition.cacheable:
        return ()
    return tuple(
        memory_key
        for memory_key, definition in MEMORY_KEYS.items()
        if definition.memory_type == memory_type and definition.cacheable
    )


def _ttl(base_seconds: int) -> int:
    base = max(1, int(base_seconds))
    ratio = max(0.0, min(0.5, float(settings.MEMORY_CACHE_TTL_JITTER_RATIO)))
    return max(1, int(base * (1 + random.uniform(-ratio, ratio))))


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _decode(
    raw: str,
    *,
    expected_key: str,
    expected_type: str,
) -> CachedMemoryEntry | bool:
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _CACHE_SCHEMA_VERSION
    ):
        raise ValueError("memory_cache_schema_invalid")
    if payload.get("kind") == "missing":
        return False
    if (
        payload.get("kind") != "record"
        or payload.get("memory_key") != expected_key
        or payload.get("memory_type") != expected_type
    ):
        raise ValueError("memory_cache_scope_invalid")
    expires_at = _parse_datetime(payload.get("expires_at"))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        raise ValueError("memory_cache_record_expired")
    return CachedMemoryEntry(
        memory_key=expected_key,
        memory_type=expected_type,
        value=payload.get("value"),
        version=int(payload["version"]),
        expires_at=expires_at,
    )


class MemoryCache:
    """Redis 仅作可丢失加速层，任何异常都由调用方回源 PostgreSQL。"""

    @staticmethod
    async def get_many(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
        memory_keys: Iterable[str],
    ) -> CacheLookup:
        requested = tuple(dict.fromkeys(memory_keys))
        if (
            not requested
            or not settings.MEMORY_CACHE_ENABLED
            or not _cacheable_keys(memory_type)
        ):
            return CacheLookup({}, requested, False)
        client = get_redis_client()
        if client is None:
            return CacheLookup({}, requested, False)
        try:
            keys = [
                _cache_key(
                    tenant_id,
                    user_id,
                    memory_type,
                    memory_key,
                )
                for memory_key in requested
            ]
            values = await client.mget(keys)
        except Exception as exc:
            logger.warning(
                "记忆缓存读取失败，回源 PostgreSQL: {}",
                type(exc).__name__,
            )
            return CacheLookup({}, requested, False)

        entries: dict[str, CachedMemoryEntry] = {}
        misses: list[str] = []
        for memory_key, raw in zip(requested, values, strict=False):
            if raw is None:
                misses.append(memory_key)
                continue
            try:
                decoded = _decode(
                    raw,
                    expected_key=memory_key,
                    expected_type=memory_type,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                misses.append(memory_key)
                continue
            if decoded is not False:
                entries[memory_key] = decoded
        if len(values) < len(requested):
            misses.extend(requested[len(values):])
        return CacheLookup(entries, tuple(misses), True)

    @staticmethod
    async def set_loaded(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
        requested_keys: Iterable[str],
        entries: Iterable[CachedMemoryEntry],
    ) -> None:
        if not settings.MEMORY_CACHE_ENABLED:
            return
        client = get_redis_client()
        if client is None:
            return
        entry_map = {entry.memory_key: entry for entry in entries}
        cacheable = set(_cacheable_keys(memory_type))
        writes = []
        for memory_key in dict.fromkeys(requested_keys):
            if memory_key not in cacheable:
                continue
            entry = entry_map.get(memory_key)
            if entry is None:
                payload = {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "kind": "missing",
                }
                ttl_seconds = _ttl(settings.MEMORY_CACHE_NEGATIVE_TTL_SEC)
            else:
                payload = {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "kind": "record",
                    "memory_key": entry.memory_key,
                    "memory_type": entry.memory_type,
                    "value": entry.value,
                    "version": entry.version,
                    "expires_at": (
                        entry.expires_at.isoformat()
                        if entry.expires_at is not None
                        else None
                    ),
                }
                ttl_seconds = _ttl(settings.MEMORY_CACHE_TTL_SEC)

            async def write(
                key: str = memory_key,
                serialized: str = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                ttl: int = ttl_seconds,
            ) -> None:
                try:
                    await client.set(
                        _cache_key(
                            tenant_id,
                            user_id,
                            memory_type,
                            key,
                        ),
                        serialized,
                        ttl_seconds=ttl,
                    )
                except Exception as exc:
                    logger.warning(
                        "记忆缓存写入失败，忽略缓存错误: {}",
                        type(exc).__name__,
                    )

            writes.append(write())
        if writes:
            await asyncio.gather(*writes)

    @staticmethod
    async def acquire_fill_lock(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
    ) -> str | None:
        client = get_redis_client()
        if client is None or not settings.MEMORY_CACHE_ENABLED:
            return None
        token = uuid4().hex
        try:
            acquired = await client.acquire_lock(
                _lock_key(tenant_id, user_id, memory_type),
                token,
                ttl_seconds=max(1, settings.MEMORY_CACHE_LOCK_TTL_SEC),
            )
        except Exception:
            return None
        return token if acquired else None

    @staticmethod
    async def release_fill_lock(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
        token: str,
    ) -> None:
        client = get_redis_client()
        if client is None:
            return
        try:
            await client.release_lock(
                _lock_key(tenant_id, user_id, memory_type),
                token,
            )
        except Exception:
            pass

    @staticmethod
    async def wait_for_fill() -> None:
        wait_seconds = max(0, settings.MEMORY_CACHE_LOCK_WAIT_MS) / 1000
        if wait_seconds > 0:
            await asyncio.sleep(min(wait_seconds, 0.1))

    @staticmethod
    async def invalidate_user(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str = "preference",
        raise_on_error: bool = False,
    ) -> bool:
        """写事务提交后清理该用户可缓存字段；失败不回滚权威数据。"""
        client = get_redis_client()
        if client is None:
            return True
        keys = [
            _cache_key(
                tenant_id,
                user_id,
                memory_type,
                memory_key,
            )
            for memory_key in _cacheable_keys(memory_type)
        ]
        if not keys:
            return True
        results = await asyncio.gather(
            *(client.delete(key) for key in keys),
            return_exceptions=True,
        )
        if any(isinstance(result, Exception) for result in results):
            logger.warning("用户记忆缓存清理不完整，后续同步动作将绕过旧缓存")
            if raise_on_error:
                raise RuntimeError("memory_cache_invalidation_failed")
            return False
        return True


__all__ = [
    "CacheLookup",
    "CachedMemoryEntry",
    "MemoryCache",
]
