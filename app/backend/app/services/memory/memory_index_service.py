"""Outbox 驱动的长期记忆索引同步。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.memory.memory_record import MemoryRecord
from app.services.memory.memory_store import delete_memory_index, upsert_memory_index


class MemoryIndexService:
    @staticmethod
    async def upsert_from_event(payload: dict[str, Any]) -> None:
        memory_id = str(payload["memory_id"])
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
            return
        # 以数据库当前版本为准，重复消费旧事件不会覆盖新版本。
        if record.version < int(payload.get("version") or record.version):
            return
        await upsert_memory_index(
            memory_id=record.id,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            memory_type=record.memory_type,
            memory_key=record.memory_key,
            value=(record.value_json or {}).get("value"),
            search_text=record.search_text,
            version=record.version,
        )

    @staticmethod
    async def delete_from_event(payload: dict[str, Any]) -> None:
        await delete_memory_index(
            memory_id=str(payload["memory_id"]),
            tenant_id=str(payload["tenant_id"]),
            user_id=int(payload["user_id"]),
            memory_type=str(payload.get("memory_type") or "preference"),
        )
