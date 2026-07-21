"""Checkpoint 注册、TTL 和归档。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from agents.checkpoint import delete_thread_checkpoint
from app.core.database import AsyncSessionLocal
from app.models.checkpoint_registry import CheckpointRegistry


class CheckpointRegistryService:
    @staticmethod
    async def record(
        *, conversation_id: int, user_id: int, thread_id: str,
        checkpoint_id: str | None, ttl_seconds: int = 30 * 24 * 3600
    ) -> None:
        async with AsyncSessionLocal() as db:
            row = await db.scalar(
                select(CheckpointRegistry).where(
                    CheckpointRegistry.conversation_id == conversation_id
                )
            )
            values = {
                "user_id": user_id,
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "status": "active",
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                "archived_at": None,
            }
            if row is None:
                db.add(CheckpointRegistry(conversation_id=conversation_id, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            await db.commit()

    @staticmethod
    async def archive_expired(*, limit: int = 100) -> int:
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(CheckpointRegistry).where(
                    CheckpointRegistry.status == "active",
                    CheckpointRegistry.expires_at <= now,
                ).limit(limit)
            )).scalars().all()
            for row in rows:
                await delete_thread_checkpoint(row.conversation_id, user_id=row.user_id)
                row.status = "archived"
                row.archived_at = now
            await db.commit()
            return len(rows)

