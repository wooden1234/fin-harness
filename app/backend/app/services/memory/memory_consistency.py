"""SQL 权威记录与向量索引的一致性扫描及 Outbox 修复。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.persistence.outbox_event import OutboxEvent
from app.services.memory.memory_catalog import MEMORY_TYPES
from app.services.memory.memory_metrics import increment
from app.services.memory.memory_service import MemoryService
from app.services.memory.memory_store import (
    get_memory_store,
    list_memory_index_entries,
)


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    tenant_id: str
    user_id: int
    memory_type: str
    missing_ids: tuple[str, ...] = ()
    stale_ids: tuple[str, ...] = ()
    orphan_ids: tuple[str, ...] = ()
    repaired_events: int = 0


class MemoryConsistencyScanner:
    """扫描结果可重复执行，修复事件使用稳定 key 保证入队幂等。"""

    @staticmethod
    async def _ensure_event(
        db,
        *,
        event_key: str,
        event_type: str,
        aggregate_id: str,
        payload: dict[str, Any],
    ) -> bool:
        existing = await db.scalar(
            select(OutboxEvent).where(OutboxEvent.event_key == event_key)
        )
        if existing is not None:
            if existing.status in {"published", "dead"}:
                existing.status = "pending"
                existing.attempts = 0
                existing.last_error = None
            return False
        db.add(
            OutboxEvent(
                event_key=event_key,
                event_type=event_type,
                aggregate_id=aggregate_id,
                payload=payload,
                status="pending",
            )
        )
        return True

    @classmethod
    async def scan(
        cls,
        *,
        tenant_id: str,
        user_id: int,
        memory_type: str = "episodic",
        repair: bool = True,
    ) -> ConsistencyReport:
        definition = MEMORY_TYPES.get(memory_type)
        if definition is None or not definition.vectorizable:
            raise ValueError(f"memory_type_not_vectorizable:{memory_type}")
        if get_memory_store() is None:
            return ConsistencyReport(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type=memory_type,
            )

        records = await MemoryService.list_active_vectorizable(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=memory_type,
        )
        sql_by_id = {str(record.id): record for record in records}
        vector_entries = await list_memory_index_entries(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=memory_type,
        )
        vector_by_id = {memory_id: version for memory_id, version in vector_entries}
        missing_ids = tuple(
            memory_id for memory_id in sql_by_id if memory_id not in vector_by_id
        )
        stale_ids = tuple(
            memory_id
            for memory_id, vector_version in vector_by_id.items()
            if memory_id in sql_by_id
            and (
                vector_version is None
                or vector_version != int(sql_by_id[memory_id].version)
            )
        )
        orphan_ids = tuple(memory_id for memory_id in vector_by_id if memory_id not in sql_by_id)

        increment("memory_consistency_scans_total")
        increment("memory_consistency_missing_total", len(missing_ids))
        increment("memory_consistency_stale_total", len(stale_ids))
        increment("memory_consistency_orphan_total", len(orphan_ids))

        repaired_events = 0
        if repair and (missing_ids or stale_ids or orphan_ids):
            async with AsyncSessionLocal() as db:
                for memory_id in (*missing_ids, *stale_ids):
                    record = sql_by_id[memory_id]
                    payload = {
                        "memory_id": memory_id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": memory_type,
                        "version": int(record.version),
                        "source": "consistency_scan",
                    }
                    repaired_events += await cls._ensure_event(
                        db,
                        event_key=(
                            f"memory:index:repair:{memory_id}:v:"
                            f"{record.version}"
                        ),
                        event_type="memory.index.upsert",
                        aggregate_id=memory_id,
                        payload=payload,
                    )
                for memory_id in orphan_ids:
                    payload = {
                        "memory_id": memory_id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "memory_type": memory_type,
                        "source": "consistency_scan",
                    }
                    repaired_events += await cls._ensure_event(
                        db,
                        event_key=f"memory:index:repair-delete:{memory_id}",
                        event_type="memory.index.delete",
                        aggregate_id=memory_id,
                        payload=payload,
                    )
                await db.commit()
        increment("memory_consistency_repair_events_total", repaired_events)
        return ConsistencyReport(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=memory_type,
            missing_ids=missing_ids,
            stale_ids=stale_ids,
            orphan_ids=orphan_ids,
            repaired_events=repaired_events,
        )

    @classmethod
    async def scan_all(
        cls,
        *,
        tenant_id: str,
        user_id: int,
        repair: bool = True,
    ) -> tuple[ConsistencyReport, ...]:
        """扫描目录中全部可向量化类型，供定时任务按用户批量调用。"""
        reports: list[ConsistencyReport] = []
        for memory_type, definition in MEMORY_TYPES.items():
            if definition.vectorizable:
                reports.append(
                    await cls.scan(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        memory_type=memory_type,
                        repair=repair,
                    )
                )
        return tuple(reports)


__all__ = ["ConsistencyReport", "MemoryConsistencyScanner"]
