"""Outbox 驱动的长期记忆索引同步。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.memory.memory_record import MemoryRecord
from app.services.memory.memory_catalog import MEMORY_TYPES
from app.services.memory.memory_audit import MemoryAuditContext, record_audit
from app.services.memory.memory_metrics import increment
from app.services.memory.memory_store import delete_memory_index, upsert_memory_index


class MemoryIndexService:
    @staticmethod
    def _audit_context(payload: dict[str, Any]) -> MemoryAuditContext:
        return MemoryAuditContext(
            tenant_id=str(payload["tenant_id"]),
            user_id=int(payload["user_id"]),
            agent_id=payload.get("agent_id"),
            task_id=payload.get("task_id"),
            trace_id=payload.get("trace_id"),
        )

    @staticmethod
    async def upsert_from_event(payload: dict[str, Any]) -> None:
        memory_id = str(payload["memory_id"])
        payload_memory_type = str(payload.get("memory_type") or "")
        payload_type_definition = MEMORY_TYPES.get(payload_memory_type)
        if (
            payload_memory_type
            and (
                payload_type_definition is None
                or not payload_type_definition.vectorizable
            )
        ):
            increment("memory_vector_upsert_rejected_total")
            await record_audit(
                context=MemoryIndexService._audit_context(payload),
                action="memory.reject",
                resource_id=memory_id,
                memory_type=payload_memory_type or None,
                details={"reason": "memory_type_not_vectorizable"},
            )
            return
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                select(MemoryRecord).where(
                    MemoryRecord.id == memory_id,
                    MemoryRecord.tenant_id == str(payload["tenant_id"]),
                    MemoryRecord.user_id == int(payload["user_id"]),
                )
            )
        if record is None or record.status != "active":
            await delete_memory_index(
                memory_id=memory_id,
                tenant_id=str(payload["tenant_id"]),
                user_id=int(payload["user_id"]),
            )
            await record_audit(
                context=MemoryIndexService._audit_context(payload),
                action="memory.vector.delete",
                resource_id=memory_id,
                memory_type=str(payload.get("memory_type") or ""),
                details={"reason": "record_missing_or_inactive"},
            )
            return
        memory_type = MEMORY_TYPES.get(record.memory_type)
        if memory_type is None or not memory_type.vectorizable:
            # preference 等结构化类型只保留 SQL/Redis 精确查询。
            increment("memory_vector_upsert_rejected_total")
            await record_audit(
                context=MemoryIndexService._audit_context(payload),
                action="memory.reject",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"reason": "memory_type_not_vectorizable"},
            )
            return
        # 以数据库当前版本为准，重复消费旧事件不会覆盖新版本。
        if record.version < int(payload.get("version") or record.version):
            return
        await upsert_memory_index(
            memory_id=record.id,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            memory_type=record.memory_type,
            search_text=record.search_text,
            version=int(record.version),
        )
        await record_audit(
            context=MemoryIndexService._audit_context(payload),
            action="memory.vector.upsert",
            resource_id=record.id,
            memory_type=record.memory_type,
            memory_key=record.memory_key,
            details={"version": int(record.version)},
        )

    @staticmethod
    async def delete_from_event(payload: dict[str, Any]) -> None:
        await delete_memory_index(
            memory_id=str(payload["memory_id"]),
            tenant_id=str(payload["tenant_id"]),
            user_id=int(payload["user_id"]),
            memory_type=str(payload.get("memory_type") or "preference"),
        )
        await record_audit(
            context=MemoryIndexService._audit_context(payload),
            action="memory.vector.delete",
            resource_id=str(payload["memory_id"]),
            memory_type=str(payload.get("memory_type") or "preference"),
        )
