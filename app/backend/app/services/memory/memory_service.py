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
from app.services.memory.memory_policy import validate_preference
from app.services.memory.memory_conflict import values_conflict
from app.core.config import settings


class MemoryService:
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
        changed = [row for row in rows if row.status in {"active", "pending"}]
        deleted = [
            row.id
            for row in rows
            if row.status in {"revoked", "expired", "rejected", "superseded", "deleted"}
        ]
        next_cursor = max((row.updated_at for row in rows if row.updated_at), default=since)
        return changed, deleted, next_cursor

    @staticmethod
    async def expire_due(*, limit: int = 100) -> int:
        """标记过期记录并生成删除索引事件。"""
        now = datetime.now(timezone.utc)
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
                record.status = "expired"
                record.updated_at = now
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
    ) -> MemoryRecord:
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
                if values_conflict((current.value_json or {}).get("value"), value):
                    return await MemoryService._create_pending_record(
                        db=db,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        memory_key=memory_key,
                        value=value,
                        provenance={
                            **(provenance or {}),
                            "conflict_with": current.id,
                        },
                        actor_id=actor_id,
                        event_type="conflict_detected",
                    )
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
                confidence=1.0,
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
            event_type = "memory.index.upsert"
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:upsert:{record.id}:v:{record.version}",
                    event_type=event_type,
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": record.memory_type,
                        "version": record.version,
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
                    event_type="activated",
                    event_key=f"memory:{record.id}:v:{record.version}:activated",
                    payload_json={"memory_key": memory_key, "value": value},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
            await db.refresh(record)
            return record

    @staticmethod
    async def _create_pending_record(
        *,
        db: Any,
        tenant_id: str,
        user_id: int,
        memory_key: str,
        value: Any,
        provenance: dict[str, Any],
        actor_id: str | None,
        event_type: str = "candidate_created",
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type="preference",
            memory_key=memory_key,
            value_json={"value": value},
            display_text=f"{memory_key}={value}",
            search_text=f"{memory_key} {value}",
            provenance_json=provenance,
            confidence=0.7,
            consent_status="pending",
            status="pending",
            version=1,
        )
        db.add(record)
        # pending 记录同样被事件引用，必须先完成 flush 再插入审计事件。
        await db.flush()
        db.add(
            MemoryEvent(
                id=str(uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=record.id,
                event_type=event_type,
                event_key=f"memory:{record.id}:{event_type}",
                payload_json={"memory_key": memory_key, "value": value},
                actor_type="system" if event_type == "conflict_detected" else "user",
                actor_id=actor_id,
            )
        )
        await db.commit()
        await db.refresh(record)
        return record

    @staticmethod
    async def create_candidate(
        *,
        tenant_id: str,
        user_id: int,
        memory_key: str,
        value: Any,
        provenance: dict[str, Any] | None = None,
        actor_id: str | None = None,
    ) -> MemoryRecord:
        validate_preference(memory_key, value)
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            existing = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "preference",
                        MemoryRecord.memory_key == memory_key,
                        MemoryRecord.status == "pending",
                    ),
                    tenant_id,
                    user_id,
                )
            )
            if existing is not None:
                return existing
            active = await db.scalar(
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
            if active is not None:
                return active
            record = await MemoryService._create_pending_record(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_key=memory_key,
                value=value,
                provenance=provenance or {"source_type": "candidate_extractor"},
                actor_id=actor_id,
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
    ) -> MemoryRecord:
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
                    },
                    status="pending",
                )
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
    async def list_candidates(*, tenant_id: str, user_id: int) -> list[MemoryRecord]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.memory_type == "preference",
                        MemoryRecord.status == "pending",
                    ),
                    tenant_id,
                    user_id,
                ).order_by(MemoryRecord.created_at.desc())
            )
            return list(result.scalars().all())

    @staticmethod
    async def decide_candidate(
        *,
        tenant_id: str,
        user_id: int,
        memory_id: str,
        decision: str,
        actor_id: str,
    ) -> MemoryRecord | None:
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                MemoryService._scope(
                    select(MemoryRecord).where(
                        MemoryRecord.id == memory_id,
                        MemoryRecord.status == "pending",
                    ),
                    tenant_id,
                    user_id,
                )
            )
            if record is None:
                return None
            now = datetime.now(timezone.utc)
            if decision == "confirm":
                record.status = "active"
                record.consent_status = "granted"
                record.consented_at = now
                db.add(
                    OutboxEvent(
                        event_key=f"memory:index:upsert:{record.id}:v:{record.version}",
                        event_type="memory.index.upsert",
                        aggregate_id=record.id,
                        payload={
                            "memory_id": record.id,
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                            "memory_type": record.memory_type,
                            "version": record.version,
                        },
                        status="pending",
                    )
                )
                event_type = "activated"
            else:
                record.status = "rejected"
                record.consent_status = "denied"
                event_type = "rejected"
            record.updated_at = now
            db.add(
                MemoryEvent(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=record.id,
                    event_type=event_type,
                    event_key=f"memory:{record.id}:{event_type}",
                    payload_json={"decision": decision},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
            await db.refresh(record)
            return record

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
    ) -> MemoryRecord | None:
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
                return None
            if expected_version is not None and record.version != expected_version:
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
                    event_key=f"memory:index:upsert:{record.id}:v:{record.version}",
                    event_type="memory.index.upsert",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": record.memory_type,
                        "version": record.version,
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
            await db.commit()
            await db.refresh(record)
            return record

    @staticmethod
    async def revoke(*, tenant_id: str, user_id: int, memory_id: str, actor_id: str | None = None) -> bool:
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
                    event_type="consent_withdrawn",
                    event_key=f"memory:{record.id}:v:{record.version}:revoked",
                    payload_json={"memory_key": record.memory_key},
                    actor_type="user",
                    actor_id=actor_id,
                )
            )
            await db.commit()
            return True
