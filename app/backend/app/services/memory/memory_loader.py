"""统一只读 Memory Loader：精确查询、白名单校验与 Projection 隔离。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from app.services.memory.memory_cache import CachedMemoryEntry, MemoryCache
from app.services.memory.memory_audit import (
    MemoryAuditContext,
    context_for,
    record_audit,
)
from app.services.memory.memory_catalog import MEMORY_KEYS, MEMORY_TYPES
from app.services.memory.memory_metrics import increment
from app.services.memory.memory_service import MemoryService
from app.services.memory.memory_store import search_memory_ids


@dataclass(frozen=True, slots=True, init=False)
class MemoryProjectionEntry:
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
        object.__setattr__(self, "_value", deepcopy(value))

    @property
    def value(self) -> Any:
        """每次返回独立副本，避免调用方修改 Loader 快照。"""
        return deepcopy(self._value)


@dataclass(frozen=True, slots=True)
class MemoryProjection:
    tenant_id: str
    user_id: int
    memory_type: str
    entries: tuple[MemoryProjectionEntry, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.memory_key for entry in self.entries)

    def as_dict(self) -> dict[str, Any]:
        """生成可安全写入 Agent State 的独立值副本。"""
        return {
            entry.memory_key: entry.value
            for entry in self.entries
        }


@dataclass(frozen=True, slots=True, init=False)
class SemanticMemoryEntry:
    memory_id: str
    memory_type: str
    memory_key: str
    display_text: str
    version: int
    expires_at: datetime | None
    _value: Any = field(repr=False)

    def __init__(
        self,
        *,
        memory_id: str,
        memory_type: str,
        memory_key: str,
        value: Any,
        display_text: str,
        version: int,
        expires_at: datetime | None,
    ) -> None:
        object.__setattr__(self, "memory_id", memory_id)
        object.__setattr__(self, "memory_type", memory_type)
        object.__setattr__(self, "memory_key", memory_key)
        object.__setattr__(self, "display_text", display_text)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "_value", deepcopy(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "value": deepcopy(self._value),
            "display_text": self.display_text,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class SemanticMemoryProjection:
    tenant_id: str
    user_id: int
    memory_type: str
    entries: tuple[SemanticMemoryEntry, ...]

    def as_list(self) -> list[dict[str, Any]]:
        """只暴露完成 SQL 核验后的候选内容。"""
        return [entry.as_dict() for entry in self.entries]


def _trusted_scope(tenant_id: str, user_id: int) -> tuple[str, int]:
    normalized_tenant = str(tenant_id).strip()
    if not normalized_tenant or normalized_tenant == "None":
        raise ValueError("memory_loader_requires_trusted_tenant_id")
    if isinstance(user_id, bool):
        raise ValueError("memory_loader_requires_trusted_user_id")
    try:
        normalized_user = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory_loader_requires_trusted_user_id") from exc
    if normalized_user <= 0:
        raise ValueError("memory_loader_requires_trusted_user_id")
    return normalized_tenant, normalized_user


def _exact_keys(
    memory_keys: Iterable[str],
    *,
    memory_type: str,
) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(
            str(memory_key).strip()
            for memory_key in memory_keys
            if str(memory_key).strip()
        )
    )
    for memory_key in normalized:
        definition = MEMORY_KEYS.get(memory_key)
        if definition is None:
            raise ValueError(f"memory_key_not_registered:{memory_key}")
        if definition.memory_type != memory_type:
            raise ValueError(
                f"memory_key_type_mismatch:{memory_key}:{memory_type}"
            )
        if (
            definition.lifecycle != "persistent"
            or not definition.sql_writable
            or not definition.prompt_visible
        ):
            raise ValueError(f"memory_key_not_loadable:{memory_key}")
    return normalized


class MemoryLoader:
    """业务侧唯一只读入口；缓存不可用时自动回源 PostgreSQL。"""

    @staticmethod
    def _projection_entry(record: Any) -> MemoryProjectionEntry:
        value = (
            record.value
            if isinstance(record, CachedMemoryEntry)
            else (record.value_json or {}).get("value")
        )
        return MemoryProjectionEntry(
            memory_key=record.memory_key,
            memory_type=record.memory_type,
            value=value,
            version=int(record.version),
            expires_at=record.expires_at,
        )

    @staticmethod
    def _cached_entry(record: Any) -> CachedMemoryEntry:
        return CachedMemoryEntry(
            memory_key=record.memory_key,
            memory_type=record.memory_type,
            value=(record.value_json or {}).get("value"),
            version=int(record.version),
            expires_at=record.expires_at,
        )

    @staticmethod
    async def load_by_keys(
        *,
        tenant_id: str,
        user_id: int,
        memory_keys: Iterable[str],
        memory_type: str = "preference",
        bypass_cache: bool = False,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryProjection:
        trusted_tenant, trusted_user = _trusted_scope(tenant_id, user_id)
        if audit_context is not None:
            audit_context = context_for(
                tenant_id=trusted_tenant,
                user_id=trusted_user,
                audit_context=audit_context,
            )
        exact_keys = _exact_keys(memory_keys, memory_type=memory_type)
        loaded: dict[str, Any] = {}

        async def load_from_database(keys: tuple[str, ...]) -> list[Any]:
            if not keys:
                return []
            return await MemoryService.list_by_keys(
                tenant_id=trusted_tenant,
                user_id=trusted_user,
                memory_type=memory_type,
                memory_keys=keys,
            )

        if bypass_cache:
            records = await load_from_database(exact_keys)
            loaded.update({record.memory_key: record for record in records})
        else:
            lookup = await MemoryCache.get_many(
                tenant_id=trusted_tenant,
                user_id=trusted_user,
                memory_type=memory_type,
                memory_keys=exact_keys,
            )
            loaded.update(lookup.entries)
            misses = lookup.misses
            cache_write_allowed = lookup.redis_available
            lock_token = (
                await MemoryCache.acquire_fill_lock(
                    tenant_id=trusted_tenant,
                    user_id=trusted_user,
                    memory_type=memory_type,
                )
                if misses and lookup.redis_available
                else None
            )
            try:
                if lock_token is not None:
                    # 加锁后再次读取，避免等待锁期间重复回源。
                    refreshed = await MemoryCache.get_many(
                        tenant_id=trusted_tenant,
                        user_id=trusted_user,
                        memory_type=memory_type,
                        memory_keys=misses,
                    )
                    loaded.update(refreshed.entries)
                    misses = refreshed.misses
                    cache_write_allowed = refreshed.redis_available
                elif misses and lookup.redis_available:
                    await MemoryCache.wait_for_fill()
                    refreshed = await MemoryCache.get_many(
                        tenant_id=trusted_tenant,
                        user_id=trusted_user,
                        memory_type=memory_type,
                        memory_keys=misses,
                    )
                    loaded.update(refreshed.entries)
                    misses = refreshed.misses
                    cache_write_allowed = refreshed.redis_available

                records = await load_from_database(misses)
                loaded.update({record.memory_key: record for record in records})
                if misses and cache_write_allowed:
                    await MemoryCache.set_loaded(
                        tenant_id=trusted_tenant,
                        user_id=trusted_user,
                        memory_type=memory_type,
                        requested_keys=misses,
                        entries=(
                            MemoryLoader._cached_entry(record)
                            for record in records
                        ),
                    )
            finally:
                if lock_token is not None:
                    await MemoryCache.release_fill_lock(
                        tenant_id=trusted_tenant,
                        user_id=trusted_user,
                        memory_type=memory_type,
                        token=lock_token,
                    )

        entries = tuple(
            MemoryLoader._projection_entry(loaded[memory_key])
            for memory_key in exact_keys
            if memory_key in loaded
        )
        projection = MemoryProjection(
            tenant_id=trusted_tenant,
            user_id=trusted_user,
            memory_type=memory_type,
            entries=entries,
        )
        if audit_context is not None:
            await record_audit(
                context=audit_context,
                action="memory.read",
                memory_type=memory_type,
                details={
                    "keys": list(exact_keys),
                    "returned_keys": list(projection.keys),
                    "cache_bypassed": bypass_cache,
                },
            )
        return projection

    @staticmethod
    async def load_for_agent(
        *,
        tenant_id: str,
        user_id: int,
        agent_id: str,
        memory_keys: Iterable[str] | None = None,
        bypass_cache: bool = False,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryProjection:
        """按 Agent Registry 静态白名单加载结构化偏好。"""
        from harness.memory_specs import get_agent_spec

        spec = get_agent_spec(agent_id)
        allowed = tuple(spec.memory_keys)
        requested = allowed if memory_keys is None else tuple(memory_keys)
        denied = sorted(set(requested) - set(allowed))
        if denied:
            if audit_context is not None:
                trusted_tenant, trusted_user = _trusted_scope(tenant_id, user_id)
                await record_audit(
                    context=context_for(
                        tenant_id=trusted_tenant,
                        user_id=trusted_user,
                        audit_context=audit_context,
                    ),
                    action="memory.reject",
                    memory_type="preference",
                    details={
                        "reason": "agent_memory_key_denied",
                        "agent_id": agent_id,
                        "denied_keys": denied,
                    },
                )
            raise PermissionError(
                f"agent_memory_key_denied:{agent_id}:{','.join(denied)}"
            )
        return await MemoryLoader.load_by_keys(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_keys=requested,
            memory_type="preference",
            bypass_cache=bypass_cache,
            audit_context=audit_context,
        )

    @staticmethod
    async def load_semantic_for_agent(
        *,
        tenant_id: str,
        user_id: int,
        agent_id: str,
        memory_type: str,
        query: str,
        top_k: int,
        audit_context: MemoryAuditContext | None = None,
    ) -> SemanticMemoryProjection:
        """按 Agent 类型白名单执行向量候选和 SQL 权威核验。"""
        from harness.memory_specs import get_agent_spec

        trusted_tenant, trusted_user = _trusted_scope(tenant_id, user_id)
        if audit_context is not None:
            audit_context = context_for(
                tenant_id=trusted_tenant,
                user_id=trusted_user,
                audit_context=audit_context,
            )
        spec = get_agent_spec(agent_id)
        if memory_type not in spec.semantic_memory_types:
            if audit_context is not None:
                await record_audit(
                    context=audit_context,
                    action="memory.reject",
                    memory_type=memory_type,
                    details={
                        "reason": "agent_semantic_memory_type_denied",
                        "agent_id": agent_id,
                    },
                )
            raise PermissionError(
                f"agent_semantic_memory_type_denied:{agent_id}:{memory_type}"
            )
        memory_type_definition = MEMORY_TYPES.get(memory_type)
        if (
            memory_type_definition is None
            or memory_type_definition.lifecycle != "persistent"
            or not memory_type_definition.vectorizable
        ):
            raise ValueError(f"memory_type_not_vectorizable:{memory_type}")

        limit = max(1, min(int(top_k), 50))
        try:
            candidate_ids = tuple(dict.fromkeys(
                str(memory_id)
                for memory_id in await search_memory_ids(
                    tenant_id=trusted_tenant,
                    user_id=trusted_user,
                    memory_type=memory_type,
                    query=query,
                    limit=limit,
                )
            ))[:limit]
        except Exception:
            increment("memory_vector_search_error_total")
            candidate_ids = ()
        increment("memory_vector_search_requests_total")
        increment("memory_vector_candidates_total", len(candidate_ids))

        (
            records,
            cross_tenant_ids,
        ) = await MemoryService.list_vector_candidates_for_verification(
            tenant_id=trusted_tenant,
            user_id=trusted_user,
            memory_type=memory_type,
            memory_ids=candidate_ids,
        )
        records_by_id = {str(record.id): record for record in records}
        cross_tenant_id_set = set(cross_tenant_ids)
        now = datetime.now(timezone.utc)
        accepted: list[SemanticMemoryEntry] = []
        stale_rejected = 0
        expired_rejected = 0
        cross_tenant_rejected = 0
        for memory_id in candidate_ids:
            if memory_id in cross_tenant_id_set:
                cross_tenant_rejected += 1
                continue
            record = records_by_id.get(memory_id)
            if record is None:
                stale_rejected += 1
                continue
            expires_at = record.expires_at
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if record.status == "expired" or (
                expires_at is not None and expires_at <= now
            ):
                expired_rejected += 1
                continue
            if (
                record.memory_type != memory_type
                or record.status != "active"
            ):
                stale_rejected += 1
                continue
            accepted.append(
                SemanticMemoryEntry(
                    memory_id=str(record.id),
                    memory_type=record.memory_type,
                    memory_key=record.memory_key,
                    value=record.value_json,
                    display_text=record.display_text,
                    version=int(record.version),
                    expires_at=expires_at,
                )
            )

        increment("memory_vector_hits_total", len(accepted))
        increment("memory_vector_stale_rejected_total", stale_rejected)
        increment("memory_vector_expired_rejected_total", expired_rejected)
        increment(
            "memory_vector_cross_tenant_rejected_total",
            cross_tenant_rejected,
        )
        projection = SemanticMemoryProjection(
            tenant_id=trusted_tenant,
            user_id=trusted_user,
            memory_type=memory_type,
            entries=tuple(accepted),
        )
        if audit_context is not None:
            await record_audit(
                context=audit_context,
                action="memory.read",
                memory_type=memory_type,
                details={
                    "semantic": True,
                    "candidate_count": len(candidate_ids),
                    "returned_count": len(projection.entries),
                },
            )
        return projection


__all__ = [
    "MemoryLoader",
    "MemoryProjection",
    "MemoryProjectionEntry",
    "SemanticMemoryEntry",
    "SemanticMemoryProjection",
]
