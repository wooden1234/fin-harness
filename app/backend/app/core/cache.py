"""共享异步 Cache-Aside 辅助层：统一信封、TTL jitter、指标与故障回源。"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import random
import re
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis_client import get_redis_client
from app.core.redis_keys import RedisKey

logger = get_logger(service="cache")

CACHE_SCHEMA_VERSION = 1
CacheKind = Literal["record", "missing"]
_DOMAIN_EVENTS = (
    "hit",
    "negative_hit",
    "miss",
    "fallback",
    "write_fail",
    "deser_fail",
    "payload_too_large",
    "fill_lock_acquired",
    "fill_lock_contended",
)
_WHITESPACE_RE = re.compile(r"\s+")
_metrics_lock = Lock()
_domain_counters: Counter[str] = Counter()


@dataclass(frozen=True, slots=True)
class CacheResult:
    """统一缓存信封。"""

    data_type: str
    kind: CacheKind
    cached_at: datetime
    payload: Any = None
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_type": self.data_type,
            "kind": self.kind,
            "cached_at": self.cached_at.astimezone(timezone.utc).isoformat(),
            "payload": self.payload if self.kind == "record" else {},
        }


def normalize_cache_text(value: str) -> str:
    """strip 并折叠连续空白。"""
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def cache_ttl_seconds(base_seconds: int) -> int:
    """应用全局 jitter，最终 TTL 至少 1 秒。"""
    base = max(1, int(base_seconds))
    ratio = max(0.0, min(0.5, float(settings.CACHE_TTL_JITTER_RATIO)))
    return max(1, int(base * (1 + random.uniform(-ratio, ratio))))


def record_cache_event(domain: str, event: str) -> None:
    """记录固定低基数 domain 指标。"""
    safe_domain = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(domain).strip().lower()
    ) or "unknown"
    safe_event = event if event in _DOMAIN_EVENTS else "fallback"
    with _metrics_lock:
        _domain_counters[f"{safe_domain}:{safe_event}"] += 1


def cache_metrics_snapshot() -> dict[str, Any]:
    """供 /api/agent/metrics 暴露的 cache.domains 快照。"""
    domains: dict[str, dict[str, int]] = {}
    with _metrics_lock:
        items = list(_domain_counters.items())
    for key, count in items:
        domain, _, event = key.partition(":")
        bucket = domains.setdefault(
            domain,
            {name: 0 for name in _DOMAIN_EVENTS},
        )
        if event in bucket:
            bucket[event] = int(count)
    return {"domains": domains}


def reset_cache_metrics() -> None:
    """仅供测试隔离进程内指标。"""
    with _metrics_lock:
        _domain_counters.clear()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def encode_cache_result(result: CacheResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":"))


def decode_cache_result(
    raw: str,
    *,
    expected_data_type: str,
) -> CacheResult:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("cache_envelope_not_object")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache_schema_invalid")
    kind = payload.get("kind")
    if kind not in {"record", "missing"}:
        raise ValueError("cache_kind_invalid")
    if payload.get("data_type") != expected_data_type:
        raise ValueError("cache_data_type_invalid")
    cached_at_raw = payload.get("cached_at")
    if not cached_at_raw:
        raise ValueError("cache_cached_at_missing")
    cached_at = datetime.fromisoformat(str(cached_at_raw))
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    body = payload.get("payload")
    if kind == "missing":
        body = {}
    elif body is None:
        raise ValueError("cache_payload_missing")
    return CacheResult(
        data_type=expected_data_type,
        kind=kind,  # type: ignore[arg-type]
        cached_at=cached_at.astimezone(timezone.utc),
        payload=body,
    )


async def cache_get(
    key: RedisKey,
    *,
    domain: str,
    data_type: str,
    enabled: bool,
) -> CacheResult | None:
    """读取并校验缓存；失败时计入指标并返回 None。"""
    if not enabled:
        return None
    client = get_redis_client()
    if client is None:
        record_cache_event(domain, "fallback")
        return None
    try:
        raw = await client.get(key)
    except Exception as exc:
        record_cache_event(domain, "fallback")
        logger.warning(
            "缓存读取失败 domain={} op=get error={}",
            domain,
            type(exc).__name__,
        )
        return None
    if raw is None:
        record_cache_event(domain, "miss")
        return None
    try:
        result = decode_cache_result(raw, expected_data_type=data_type)
    except (TypeError, ValueError, json.JSONDecodeError, KeyError):
        record_cache_event(domain, "deser_fail")
        try:
            await client.delete(key)
        except Exception:
            pass
        return None
    if result.kind == "missing":
        record_cache_event(domain, "negative_hit")
    else:
        record_cache_event(domain, "hit")
    return result


async def cache_set(
    key: RedisKey,
    result: CacheResult,
    *,
    domain: str,
    ttl_seconds: int,
    enabled: bool,
    max_bytes: int | None = None,
) -> bool:
    """写入缓存；超限或失败时不阻断业务。"""
    if not enabled:
        return False
    client = get_redis_client()
    if client is None:
        record_cache_event(domain, "fallback")
        return False
    try:
        serialized = encode_cache_result(result)
    except (TypeError, ValueError) as exc:
        record_cache_event(domain, "write_fail")
        logger.warning(
            "缓存序列化失败 domain={} op=set error={}",
            domain,
            type(exc).__name__,
        )
        return False
    if max_bytes is not None and len(serialized.encode("utf-8")) > max_bytes:
        record_cache_event(domain, "payload_too_large")
        return False
    try:
        await client.set(
            key,
            serialized,
            ttl_seconds=cache_ttl_seconds(ttl_seconds),
        )
        return True
    except Exception as exc:
        record_cache_event(domain, "write_fail")
        logger.warning(
            "缓存写入失败 domain={} op=set error={}",
            domain,
            type(exc).__name__,
        )
        return False


async def cache_delete(
    key: RedisKey,
    *,
    domain: str,
    enabled: bool = True,
) -> None:
    """删除缓存；失败只记日志。"""
    if not enabled:
        return
    client = get_redis_client()
    if client is None:
        return
    try:
        await client.delete(key)
    except Exception as exc:
        logger.warning(
            "缓存删除失败 domain={} op=delete error={}",
            domain,
            type(exc).__name__,
        )


async def cache_fill_lock(
    lock_key: RedisKey,
    *,
    domain: str,
    ttl_seconds: int,
    enabled: bool,
) -> str | None:
    """尝试获取填充锁；争用时返回 None。"""
    if not enabled:
        return None
    client = get_redis_client()
    if client is None:
        return None
    token = uuid4().hex
    try:
        acquired = await client.acquire_lock(
            lock_key,
            token,
            ttl_seconds=max(1, int(ttl_seconds)),
        )
    except Exception:
        return None
    if acquired:
        record_cache_event(domain, "fill_lock_acquired")
        return token
    record_cache_event(domain, "fill_lock_contended")
    return None


async def cache_release_fill_lock(
    lock_key: RedisKey,
    token: str | None,
    *,
    domain: str,
) -> None:
    if not token:
        return
    client = get_redis_client()
    if client is None:
        return
    try:
        await client.release_lock(lock_key, token)
    except Exception as exc:
        logger.warning(
            "缓存解锁失败 domain={} op=lock_release error={}",
            domain,
            type(exc).__name__,
        )


async def with_fill_lock(
    *,
    cache_key: RedisKey,
    lock_key: RedisKey,
    domain: str,
    data_type: str,
    enabled: bool,
    lock_ttl_seconds: int,
    lock_wait_ms: int,
    loader: Callable[[], Awaitable[Any]],
    writer: Callable[[Any], Awaitable[None]],
    reader: Callable[[], Awaitable[CacheResult | None]] | None = None,
) -> Any:
    """带短租约 fill lock 的加载流程；争用方等待后重读，仍 miss 则直调 loader。"""

    async def _read() -> CacheResult | None:
        if reader is not None:
            return await reader()
        return await cache_get(
            cache_key,
            domain=domain,
            data_type=data_type,
            enabled=enabled,
        )

    existing = await _read()
    if existing is not None and existing.kind == "record":
        return existing.payload

    token = await cache_fill_lock(
        lock_key,
        domain=domain,
        ttl_seconds=lock_ttl_seconds,
        enabled=enabled,
    )
    try:
        if token is None and enabled:
            await asyncio.sleep(max(0.0, lock_wait_ms) / 1000.0)
            raced = await _read()
            if raced is not None and raced.kind == "record":
                return raced.payload
        value = await loader()
        if token is not None:
            await writer(value)
        return value
    finally:
        await cache_release_fill_lock(lock_key, token, domain=domain)


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def make_record(
    *,
    data_type: str,
    payload: Any,
) -> CacheResult:
    return CacheResult(
        data_type=data_type,
        kind="record",
        cached_at=_utc_now(),
        payload=payload,
    )


def make_missing(*, data_type: str) -> CacheResult:
    return CacheResult(
        data_type=data_type,
        kind="missing",
        cached_at=_utc_now(),
        payload={},
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheResult",
    "cache_delete",
    "cache_fill_lock",
    "cache_get",
    "cache_metrics_snapshot",
    "cache_release_fill_lock",
    "cache_set",
    "cache_ttl_seconds",
    "decode_cache_result",
    "encode_cache_result",
    "is_finite_number",
    "make_missing",
    "make_record",
    "normalize_cache_text",
    "record_cache_event",
    "reset_cache_metrics",
    "with_fill_lock",
]
