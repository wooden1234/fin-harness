"""应用级异步 Redis 客户端、连接池与故障降级。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, TypeVar

import redis.asyncio as redis_async
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    TimeoutError as RedisTimeoutError,
)

from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis_keys import RedisKey
from app.core.redis_metrics import (
    record_initialization,
    record_operation,
    snapshot as metrics_snapshot,
)

logger = get_logger(service="redis")
T = TypeVar("T")


class RedisUnavailableError(RuntimeError):
    """Redis 操作经过重试后仍不可用。"""


class RedisClient:
    """限制原始客户端边界，并统一处理超时、重试和指标。"""

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        retry_attempts: int,
        retry_base_delay_ms: int,
    ) -> None:
        self._client = client
        self._retry_attempts = max(0, retry_attempts)
        self._retry_base_delay_ms = max(0, retry_base_delay_ms)
        self._available = False
        self._last_error_type: str | None = None
        self._last_success_at: datetime | None = None

    @staticmethod
    def _require_key(key: RedisKey) -> str:
        if not isinstance(key, RedisKey):
            raise TypeError("Redis 操作必须使用 RedisKeyBuilder 构造的 RedisKey")
        return key.value

    async def _execute(
        self,
        operation: str,
        callback: Callable[[], Awaitable[T]],
    ) -> T:
        started_at = perf_counter()
        retries = 0
        success = False
        try:
            for attempt in range(self._retry_attempts + 1):
                try:
                    result = await callback()
                    self._available = True
                    self._last_error_type = None
                    self._last_success_at = datetime.now(timezone.utc)
                    success = True
                    return result
                except (RedisConnectionError, RedisTimeoutError) as exc:
                    self._available = False
                    self._last_error_type = type(exc).__name__
                    if attempt >= self._retry_attempts:
                        raise RedisUnavailableError(
                            f"Redis {operation} 操作不可用"
                        ) from exc
                    retries += 1
                    delay_seconds = (
                        self._retry_base_delay_ms * (2**attempt) / 1000
                    )
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
                except RedisError:
                    self._available = False
                    self._last_error_type = "RedisError"
                    raise
            raise RedisUnavailableError(f"Redis {operation} 操作不可用")
        finally:
            record_operation(
                operation,
                success=success,
                latency_ms=(perf_counter() - started_at) * 1000,
                retries=retries,
            )

    async def ping(self) -> bool:
        return bool(await self._execute("ping", self._client.ping))

    async def get(self, key: RedisKey) -> str | None:
        raw_key = self._require_key(key)
        value = await self._execute("get", lambda: self._client.get(raw_key))
        if value is None or isinstance(value, str):
            return value
        return str(value)

    async def mget(self, keys: Iterable[RedisKey]) -> list[str | None]:
        raw_keys = [self._require_key(key) for key in keys]
        if not raw_keys:
            return []
        values = await self._execute(
            "mget",
            lambda: self._client.mget(raw_keys),
        )
        return [
            value if value is None or isinstance(value, str) else str(value)
            for value in values
        ]

    async def set(
        self,
        key: RedisKey,
        value: str,
        *,
        ttl_seconds: int,
        only_if_absent: bool = False,
    ) -> bool:
        raw_key = self._require_key(key)
        if ttl_seconds <= 0:
            raise ValueError("Redis Key 必须设置正数 TTL")
        result = await self._execute(
            "set",
            lambda: self._client.set(
                raw_key,
                value,
                ex=ttl_seconds,
                nx=only_if_absent,
            ),
        )
        return bool(result)

    async def delete(self, key: RedisKey) -> int:
        raw_key = self._require_key(key)
        return int(
            await self._execute("delete", lambda: self._client.delete(raw_key))
        )

    async def acquire_lock(
        self,
        key: RedisKey,
        token: str,
        *,
        ttl_seconds: int,
    ) -> bool:
        """只允许持有随机 token 的调用方创建有期限锁。"""
        if not token:
            raise ValueError("Redis 锁 token 不能为空")
        raw_key = self._require_key(key)
        if ttl_seconds <= 0:
            raise ValueError("Redis 锁必须设置正数 TTL")
        result = await self._execute(
            "lock_acquire",
            lambda: self._client.set(
                raw_key,
                token,
                ex=ttl_seconds,
                nx=True,
            ),
        )
        return bool(result)

    async def release_lock(self, key: RedisKey, token: str) -> bool:
        """通过 Lua 比较 token 后原子删除，避免误解锁其他请求。"""
        if not token:
            raise ValueError("Redis 锁 token 不能为空")
        raw_key = self._require_key(key)
        script = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end"
        )
        result = await self._execute(
            "lock_release",
            lambda: self._client.eval(script, 1, raw_key, token),
        )
        return bool(result)

    async def sadd(
        self,
        key: RedisKey,
        *values: str,
        ttl_seconds: int,
    ) -> int:
        raw_key = self._require_key(key)
        if not values:
            return 0
        if ttl_seconds <= 0:
            raise ValueError("Redis Key 必须设置正数 TTL")

        async def add_and_expire() -> int:
            async with self._client.pipeline(transaction=True) as pipeline:
                pipeline.sadd(raw_key, *values)
                pipeline.expire(raw_key, ttl_seconds)
                added, _ = await pipeline.execute()
                return int(added)

        return await self._execute("sadd", add_and_expire)

    async def close(self) -> None:
        try:
            await self._client.aclose()
        finally:
            self._available = False

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._available else "degraded",
            "required": settings.REDIS_REQUIRED,
            "last_error_type": self._last_error_type,
            "last_success_at": (
                self._last_success_at.isoformat()
                if self._last_success_at is not None
                else None
            ),
        }


_redis_client: RedisClient | None = None


async def init_redis() -> RedisClient | None:
    """初始化连接池；非强依赖模式下连接失败不会阻止应用启动。"""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not settings.REDIS_ENABLED:
        if settings.REDIS_REQUIRED:
            raise RuntimeError("REDIS_REQUIRED=true 时不能关闭 Redis")
        logger.info("Redis 已通过 REDIS_ENABLED=false 关闭")
        return None
    if not settings.REDIS_URL.strip():
        if settings.REDIS_REQUIRED:
            raise RuntimeError("REDIS_REQUIRED=true 时必须配置 REDIS_URL")
        logger.warning("未配置 REDIS_URL，Redis 以降级模式运行")
        record_initialization(success=False)
        return None

    raw_client = redis_async.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        max_connections=max(1, settings.REDIS_MAX_CONNECTIONS),
        socket_connect_timeout=max(0.01, settings.REDIS_CONNECT_TIMEOUT_SEC),
        socket_timeout=max(0.01, settings.REDIS_SOCKET_TIMEOUT_SEC),
        health_check_interval=max(
            0,
            settings.REDIS_HEALTH_CHECK_INTERVAL_SEC,
        ),
    )
    client = RedisClient(
        raw_client,
        retry_attempts=settings.REDIS_RETRY_ATTEMPTS,
        retry_base_delay_ms=settings.REDIS_RETRY_BASE_DELAY_MS,
    )
    _redis_client = client
    try:
        await client.ping()
        record_initialization(success=True)
        logger.info("Redis 连接池初始化成功")
    except Exception as exc:
        record_initialization(success=False)
        logger.warning(
            "Redis 初始化探测失败，应用继续以降级模式运行: {}",
            type(exc).__name__,
        )
        if settings.REDIS_REQUIRED:
            await close_redis()
            raise RuntimeError("强依赖 Redis 初始化失败") from exc
    return client


def get_redis_client() -> RedisClient | None:
    """返回受控客户端；业务层不得获取原始 redis-py 客户端。"""
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    client = _redis_client
    _redis_client = None
    if client is not None:
        try:
            await client.close()
            logger.info("Redis 连接池已关闭")
        except Exception as exc:
            logger.warning("Redis 连接池关闭失败: {}", type(exc).__name__)


async def redis_health() -> dict[str, Any]:
    if not settings.REDIS_ENABLED:
        return {
            "status": "disabled",
            "required": settings.REDIS_REQUIRED,
            "last_error_type": None,
            "last_success_at": None,
        }
    client = _redis_client
    if client is None:
        return {
            "status": "degraded",
            "required": settings.REDIS_REQUIRED,
            "last_error_type": "NotInitialized",
            "last_success_at": None,
        }
    try:
        await client.ping()
    except Exception:
        pass
    return client.status()


def redis_metrics() -> dict[str, Any]:
    return metrics_snapshot()


__all__ = [
    "RedisClient",
    "RedisUnavailableError",
    "close_redis",
    "get_redis_client",
    "init_redis",
    "redis_health",
    "redis_metrics",
]
