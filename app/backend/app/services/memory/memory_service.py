"""长期记忆 CRUD 与审计事件服务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.memory.memory_event import MemoryEvent
from app.models.memory.memory_record import MemoryRecord
from app.models.persistence.outbox_event import OutboxEvent
from app.services.memory.memory_cache import MemoryCache
from app.services.memory.memory_audit import MemoryAuditContext, add_audit, context_for
from app.services.memory.memory_catalog import MEMORY_TYPES
from app.services.memory.memory_policy import validate_preference
from app.services.memory.memory_conflict import values_conflict
from app.services.memory.memory_audit import record_audit
from app.core.config import settings


class MemoryService:
    @staticmethod
    def _event_context(context: MemoryAuditContext) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "agent_id": context.agent_id,
                "task_id": context.task_id,
                "trace_id": context.trace_id,
            }.items()
            if value
        }

    @staticmethod
    def _add_cache_invalidation(
        db,
        *,
        context: MemoryAuditContext,
        reason: str,
    ) -> None:
        db.add(
            OutboxEvent(
                event_key=(
                    f"memory:cache:invalidate:{context.tenant_id}:"
                    f"{context.user_id}:{reason}"
                ),
                event_type="memory.cache.invalidate",
                aggregate_id=f"{context.tenant_id}:{context.user_id}",
                payload={
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "memory_type": "preference",
                    **MemoryService._event_context(context),
                },
                status="pending",
            )
        )

    @staticmethod
    def _scope(statement, tenant_id: str, user_id: int):
        return statement.where(
            MemoryRecord.tenant_id == tenant_id,
            MemoryRecord.user_id == user_id,
        )

    @staticmethod
    async def list(*, tenant_id: str, user_id: int) -> list[MemoryRecord]:
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "preference",
                        MemoryRecord.status == "active",
                        (MemoryRecord.expires_at.is_(None) | (MemoryRecord.expires_at > now)),
                    ),
                    tenant_id,
                    user_id,
                ).order_by(MemoryRecord.updated_at.desc())
            )
            return list(result.scalars().all())

    @staticmethod
    async def profile(*, tenant_id: str, user_id: int) -> list[MemoryRecord]:
        """返回用户画像视图所需的 active 偏好。"""
        return await MemoryService.list(tenant_id=tenant_id, user_id=user_id)

    @staticmethod
    async def list_by_keys(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
        memory_keys: tuple[str, ...],
    ) -> list[MemoryRecord]:
        """在可信租户和用户作用域内精确查询指定 active 记忆。"""
        if not memory_keys:
            return []
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == memory_type,
                        MemoryRecord.memory_key.in_(memory_keys),
                        MemoryRecord.status == "active",
                        (
                            MemoryRecord.expires_at.is_(None)
                            | (MemoryRecord.expires_at > now)
                        ),
                    ),
                    tenant_id,
                    user_id,
                )
            )
            records = list(result.scalars().all())
        order = {memory_key: index for index, memory_key in enumerate(memory_keys)}
        return sorted(
            records,
            key=lambda record: order.get(record.memory_key, len(order)),
        )

    @staticmethod
    async def list_vector_candidates_for_verification(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
        memory_ids: tuple[str, ...],
    ) -> tuple[list[MemoryRecord], tuple[str, ...]]:
        """返回作用域内候选；跨租户候选只返回 ID，不暴露记录内容。"""
        if not memory_ids:
            return [], ()
        async with AsyncSessionLocal() as db:
            scoped_result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.id.in_(memory_ids),
                        MemoryRecord.memory_type == memory_type,
                    ),
                    tenant_id,
                    user_id,
                )
            )
            scoped_records = list(scoped_result.scalars().all())
            scoped_ids = {str(record.id) for record in scoped_records}
            unresolved_ids = tuple(
                memory_id
                for memory_id in memory_ids
                if memory_id not in scoped_ids
            )
            if not unresolved_ids:
                return scoped_records, ()
            cross_tenant_result = await db.execute(
                select(MemoryRecord.id).where(
                    MemoryRecord.id.in_(unresolved_ids),
                    (
                        (MemoryRecord.tenant_id != tenant_id)
                        | (MemoryRecord.user_id != user_id)
                    ),
                )
            )
            cross_tenant_ids = tuple(
                str(memory_id)
                for memory_id in cross_tenant_result.scalars().all()
            )
            return scoped_records, cross_tenant_ids

    @staticmethod
    async def sync(
        *,
        tenant_id: str,
        user_id: int,
        since: datetime | None = None,
        limit: int = 100,
    ) -> tuple[list[MemoryRecord], list[str], datetime | None]:
        """按更新时间返回当前用户的记忆增量和失效记录。"""
        async with AsyncSessionLocal() as db:
            statement = MemoryService._scope(
                select(MemoryRecord),
                tenant_id,
                user_id,
            )
            if since is not None:
                statement = statement.where(MemoryRecord.updated_at > since)
            rows = list(
                (
                    await db.execute(
                        statement.order_by(MemoryRecord.updated_at.asc()).limit(limit)
                    )
                ).scalars().all()
            )
        changed = [row for row in rows if row.status == "active"]
        deleted = [
            row.id
            for row in rows
            if row.status
            in {
                "pending",
                "revoked",
                "expired",
                "rejected",
                "superseded",
                "deleted",
            }
        ]
        next_cursor = max((row.updated_at for row in rows if row.updated_at), default=since)
        return changed, deleted, next_cursor

    @staticmethod
    async def expire_due(*, limit: int = 100) -> int:
        """标记过期记录并生成删除索引事件。"""
        now = datetime.now(timezone.utc)
        invalidation_scopes: set[tuple[str, int]] = set()
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MemoryRecord)
                .where(
                    MemoryRecord.status == "active",
                    MemoryRecord.expires_at.is_not(None),
                    MemoryRecord.expires_at <= now,
                )
                .limit(limit)
            )
            records = list(result.scalars().all())
            for record in records:
                context = context_for(
                    tenant_id=record.tenant_id,
                    user_id=int(record.user_id),
                )
                record.status = "expired"
                record.updated_at = now
                if record.memory_type == "preference":
                    invalidation_scopes.add(
                        (record.tenant_id, int(record.user_id))
                    )
                    MemoryService._add_cache_invalidation(
                        db,
                        context=context,
                        reason=f"expired:{record.id}:v:{record.version}",
                    )
                add_audit(
                    db,
                    context=context,
                    action="memory.expire",
                    resource_id=record.id,
                    memory_type=record.memory_type,
                    memory_key=record.memory_key,
                    details={"version": int(record.version)},
                )
                db.add(
                    OutboxEvent(
                        event_key=f"memory:index:delete:{record.id}:expired",
                        event_type="memory.index.delete",
                        aggregate_id=record.id,
                        payload={
                            "memory_id": record.id,
                            "tenant_id": record.tenant_id,
                            "user_id": record.user_id,
                            "memory_type": record.memory_type,
                        },
                        status="pending",
                    )
                )
            await db.commit()
        for tenant_id, user_id in invalidation_scopes:
            await MemoryCache.invalidate_user(
                tenant_id=tenant_id,
                user_id=user_id,
            )
        return len(records)

    @staticmethod
    async def create(
        *,
        tenant_id: str,
        user_id: int,
        memory_key: str,
        value: Any,
        display_text: str | None = None,
        ttl_days: int | None = None,
        provenance: dict[str, Any] | None = None,
        actor_id: str | None = None,
        confidence: float = 1.0,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryRecord:
        context = context_for(
            tenant_id=tenant_id,
            user_id=user_id,
            audit_context=audit_context,
        )
        rule = validate_preference(memory_key, value)
        now = datetime.now(timezone.utc)
        expires_at = (
            now + timedelta(days=ttl_days if ttl_days is not None else rule.ttl_days)
            if (ttl_days is not None or rule.ttl_days is not None)
            else None
        )
        async with AsyncSessionLocal() as db:
            current = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "preference",
                        MemoryRecord.memory_key == memory_key,
                        MemoryRecord.status == "active",
                    ),
                    tenant_id,
                    user_id,
                )
            )
            if current is not None:
                if not values_conflict(
                    (current.value_json or {}).get("value"),
                    value,
                ):
                    return current
                current.status = "superseded"
                current.updated_at = now
                db.add(
                    OutboxEvent(
                        event_key=f"memory:index:delete:{current.id}:superseded:v:{current.version}",
                        event_type="memory.index.delete",
                        aggregate_id=current.id,
                        payload={
                            "memory_id": current.id,
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                            "memory_type": current.memory_type,
                            **MemoryService._event_context(context),
                        },
                        status="pending",
                    )
                )
                db.add(
                    MemoryEvent(
                        id=str(uuid4()),
                        tenant_id=tenant_id,
                        user_id=user_id,
                        memory_id=current.id,
                        event_type="superseded",
                        event_key=f"memory:{current.id}:v:{current.version}:superseded",
                        payload_json={"superseded_by": memory_key},
                        actor_type="user",
                        actor_id=actor_id,
                    )
                )

            record = MemoryRecord(
                id=str(uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type="preference",
                memory_key=memory_key,
                value_json={"value": value},
                display_text=display_text or f"{memory_key}={value}",
                search_text=f"{memory_key} {display_text or value}",
                provenance_json=provenance or {"source_type": "explicit_user_command"},
                confidence=max(0.0, min(1.0, confidence)),
                consent_status="granted",
                consented_at=now,
                status="active",
                version=(current.version + 1 if current else 1),
                supersedes_id=current.id if current else None,
                expires_at=expires_at,
            )
            db.add(record)
            # 事件通过外键引用新记录，先 flush 确保父记录已写入当前事务。
            await db.flush()
            MemoryService._add_cache_invalidation(
                db,
                context=context,
                reason=f"write:{record.id}:v:{record.version}",
            )
            add_audit(
                db,
                context=context,
                action="memory.write",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"version": int(record.version), "source": "create"},
            )
            db.add(
                MemoryEvent(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=record.id,
                    event_type="activated",
                    event_key=f"memory:{record.id}:v:{record.version}:activated",
                    payload_json={"memory_key": memory_key, "value": value},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
            await db.refresh(record)
        await MemoryCache.invalidate_user(
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return record

    @staticmethod
    async def create_episodic(
        *,
        tenant_id: str,
        user_id: int,
        event_key: str,
        value: dict[str, Any],
        display_text: str,
        expires_at: datetime | None = None,
        source_conversation_id: int | None = None,
        source_message_id: int | None = None,
        source_run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        actor_id: str | None = None,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryRecord:
        context = context_for(
            tenant_id=tenant_id,
            user_id=user_id,
            audit_context=audit_context,
        )
        now = datetime.now(timezone.utc)
        # episodic memory 默认保留 30～90 天范围内的配置值，允许调用方显式覆盖。
        ttl_days = max(30, min(90, settings.EPISODIC_MEMORY_TTL_DAYS))
        effective_expires_at = expires_at or (now + timedelta(days=ttl_days))
        async with AsyncSessionLocal() as db:
            record = MemoryRecord(
                id=str(uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type="episodic",
                memory_key=event_key,
                value_json=value,
                display_text=display_text,
                search_text=f"{event_key} {display_text}",
                provenance_json={
                    **(provenance or {"source_type": "episodic_event"}),
                    "conversation_id": source_conversation_id,
                    "message_id": source_message_id,
                    "run_id": source_run_id,
                },
                confidence=1.0,
                consent_status="granted",
                consented_at=now,
                status="active",
                expires_at=effective_expires_at,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
                source_run_id=source_run_id,
            )
            db.add(record)
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:upsert:{record.id}:v:1",
                    event_type="memory.index.upsert",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": "episodic",
                        "version": 1,
                        **MemoryService._event_context(context),
                    },
                    status="pending",
                )
            )
            add_audit(
                db,
                context=context,
                action="memory.write",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"version": 1, "source": "create_episodic"},
            )
            await db.commit()
            await db.refresh(record)
        return record

    @staticmethod
    async def list_episodic(*, tenant_id: str, user_id: int) -> list[MemoryRecord]:
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "episodic",
                        MemoryRecord.status == "active",
                        (MemoryRecord.expires_at.is_(None) | (MemoryRecord.expires_at > now)),
                    ),
                    tenant_id,
                    user_id,
                ).order_by(MemoryRecord.created_at.desc())
            )
            return list(result.scalars().all())

    @staticmethod
    async def list_active_vectorizable(
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str,
    ) -> list[MemoryRecord]:
        """返回一致性扫描所需的 SQL 权威向量记录。"""
        definition = MEMORY_TYPES.get(memory_type)
        if definition is None or not definition.vectorizable:
            raise ValueError(f"memory_type_not_vectorizable:{memory_type}")
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == memory_type,
                        MemoryRecord.status == "active",
                        (
                            MemoryRecord.expires_at.is_(None)
                            | (MemoryRecord.expires_at > now)
                        ),
                    ),
                    tenant_id,
                    user_id,
                )
            )
            return list(result.scalars().all())


    @staticmethod
    async def update(
        *,
        tenant_id: str,
        user_id: int,
        memory_id: str,
        value: Any | None = None,
        display_text: str | None = None,
        ttl_days: int | None = None,
        expected_version: int | None = None,
        reason: str | None = None,
        actor_id: str | None = None,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryRecord | None:
        context = context_for(
            tenant_id=tenant_id,
            user_id=user_id,
            audit_context=audit_context,
        )
        if expected_version is None:
            raise ValueError("expected_version_required")
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.id == memory_id,
                        MemoryRecord.status == "active",
                    ).with_for_update(),
                    tenant_id,
                    user_id,
                )
            )
            if record is None:
                return None
            if record.version != expected_version:
                await record_audit(
                    context=context,
                    action="memory.reject",
                    resource_id=record.id,
                    memory_type=record.memory_type,
                    memory_key=record.memory_key,
                    details={
                        "reason": "expected_version_mismatch",
                        "expected_version": int(expected_version),
                        "actual_version": int(record.version),
                    },
                )
                raise ValueError("记忆版本已变化，请刷新后重试")
            next_value = value if value is not None else record.value_json["value"]
            rule = validate_preference(record.memory_key, next_value)
            record.value_json = {"value": next_value}
            record.search_text = f"{record.memory_key} {display_text or next_value}"
            if display_text is not None:
                record.display_text = display_text
            if ttl_days is not None:
                record.expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
            elif rule.ttl_days is not None:
                record.expires_at = datetime.now(timezone.utc) + timedelta(days=rule.ttl_days)
            record.version += 1
            record.updated_at = datetime.now(timezone.utc)
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:delete:{record.id}:preference:v:{record.version}",
                    event_type="memory.index.delete",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": record.memory_type,
                        **MemoryService._event_context(context),
                    },
                    status="pending",
                )
            )
            db.add(
                MemoryEvent(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=record.id,
                    event_type="updated",
                    event_key=f"memory:{record.id}:v:{record.version}:updated",
                    payload_json={
                        "memory_key": record.memory_key,
                        "value": next_value,
                        "reason": reason,
                    },
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            MemoryService._add_cache_invalidation(
                db,
                context=context,
                reason=f"update:{record.id}:v:{record.version}",
            )
            add_audit(
                db,
                context=context,
                action="memory.update",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={
                    "old_version": int(expected_version),
                    "new_version": int(record.version),
                },
            )
            await db.commit()
            await db.refresh(record)
        await MemoryCache.invalidate_user(
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return record

    @staticmethod
    async def revoke(
        *,
        tenant_id: str,
        user_id: int,
        memory_id: str,
        actor_id: str | None = None,
        audit_context: MemoryAuditContext | None = None,
    ) -> bool:
        context = context_for(
            tenant_id=tenant_id,
            user_id=user_id,
            audit_context=audit_context,
        )
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.id == memory_id,
                        MemoryRecord.status == "active",
                    ),
                    tenant_id,
                    user_id,
                )
            )
            if record is None:
                return False
            now = datetime.now(timezone.utc)
            record.status = "revoked"
            record.withdrawn_at = now
            record.updated_at = now
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:delete:{record.id}:v:{record.version}",
                    event_type="memory.index.delete",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": record.memory_type,
                        **MemoryService._event_context(context),
                    },
                    status="pending",
                )
            )
            if record.memory_type == "preference":
                MemoryService._add_cache_invalidation(
                    db,
                    context=context,
                    reason=f"delete:{record.id}:v:{record.version}",
                )
            add_audit(
                db,
                context=context,
                action="memory.delete",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"version": int(record.version)},
            )
            db.add(
                MemoryEvent(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=record.id,
                    event_type="consent_withdrawn",
                    event_key=f"memory:{record.id}:v:{record.version}:revoked",
                    payload_json={"memory_key": record.memory_key},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
        if record.memory_type == "preference":
            await MemoryCache.invalidate_user(
                tenant_id=tenant_id,
                user_id=user_id,
            )
        return True

    @staticmethod
    async def revoke_by_key(
        *,
        tenant_id: str,
        user_id: int,
        memory_key: str,
        actor_id: str | None = None,
        audit_context: MemoryAuditContext | None = None,
    ) -> bool:
        """按规则识别出的偏好 key 同步失效当前 active 版本。"""
        context = context_for(
            tenant_id=tenant_id,
            user_id=user_id,
            audit_context=audit_context,
        )
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "preference",
                        MemoryRecord.memory_key == memory_key,
                        MemoryRecord.status == "active",
                    ),
                    tenant_id,
                    user_id,
                )
            )
            if record is None:
                return False
            now = datetime.now(timezone.utc)
            record.status = "revoked"
            record.withdrawn_at = now
            record.updated_at = now
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:delete:{record.id}:v:{record.version}",
                    event_type="memory.index.delete",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": record.memory_type,
                        **MemoryService._event_context(context),
                    },
                    status="pending",
                )
            )
            MemoryService._add_cache_invalidation(
                db,
                context=context,
                reason=f"delete-key:{record.id}:v:{record.version}",
            )
            add_audit(
                db,
                context=context,
                action="memory.delete",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"version": int(record.version), "by_key": True},
            )
            db.add(
                MemoryEvent(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=record.id,
                    event_type="consent_withdrawn",
                    event_key=f"memory:{record.id}:v:{record.version}:revoked",
                    payload_json={"memory_key": record.memory_key},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
        await MemoryCache.invalidate_user(
            tenant_id=tenant_id,
            user_id=user_id,
        )
        return True
