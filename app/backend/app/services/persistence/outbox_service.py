"""Outbox 入队与幂等重试 worker。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import or_, select

from app.core.database import AsyncSessionLocal
from app.core.logger import get_logger
from app.models.persistence.outbox_event import OutboxEvent
from app.services.agent.agent_run_service import AgentRunService
from app.services.conversation.conversation_service import ConversationService
from app.services.memory.memory_index_service import MemoryIndexService
from agents.checkpoint import delete_thread_checkpoint

logger = get_logger(service="outbox")


class OutboxService:
    """管理数据库 outbox，并提供可重复执行的补偿处理。"""

    @staticmethod
    async def enqueue_assistant_persist(
        *,
        run_id: str,
        user_id: int,
        conversation_id: int,
        content: str,
        tenant_id: str = "default",
    ) -> OutboxEvent:
        event_key = f"assistant_message:{run_id}"
        payload = {
            "run_id": run_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
            "content": content,
        }
        async with AsyncSessionLocal() as db:
            existing = await db.scalar(
                select(OutboxEvent).where(OutboxEvent.event_key == event_key)
            )
            if existing is not None:
                return existing
            event = OutboxEvent(
                event_key=event_key,
                event_type="assistant_message.persist",
                aggregate_id=run_id,
                payload=payload,
                status="pending",
            )
            db.add(event)
            await db.commit()
            await db.refresh(event)
            return event

    @staticmethod
    async def _claim_one() -> OutboxEvent | None:
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(minutes=10)
        async with AsyncSessionLocal() as db:
            stmt = (
                select(OutboxEvent)
                .where(
                    or_(
                        (OutboxEvent.status == "pending")
                        & (OutboxEvent.available_at <= now),
                        (OutboxEvent.status == "processing")
                        & (OutboxEvent.locked_at <= stale_before),
                    )
                )
                .order_by(OutboxEvent.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            event = await db.scalar(stmt)
            if event is None:
                return None
            event.status = "processing"
            event.locked_at = now
            event.attempts += 1
            await db.commit()
            await db.refresh(event)
            return event

    @staticmethod
    async def _finish(event_id: str, *, success: bool, error: str = "") -> None:
        async with AsyncSessionLocal() as db:
            event = await db.get(OutboxEvent, event_id)
            if event is None:
                return
            if success:
                event.status = "published"
                event.published_at = datetime.now(timezone.utc)
                event.last_error = None
            else:
                # 指数退避并设置上限，避免故障时持续打满数据库。
                if event.attempts >= 20:
                    event.status = "dead"
                else:
                    delay = min(3600, 2 ** min(event.attempts, 10))
                    event.status = "pending"
                    event.available_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                event.last_error = error[:4000]
            event.locked_at = None
            await db.commit()

    @classmethod
    async def process_once(cls) -> bool:
        event = await cls._claim_one()
        if event is None:
            return False
        try:
            payload: dict[str, Any] = event.payload or {}
            if event.event_type == "assistant_message.persist":
                await ConversationService.save_assistant_message(
                    user_id=int(payload["user_id"]),
                    conversation_id=int(payload["conversation_id"]),
                    content=str(payload["content"]),
                    run_id=str(payload["run_id"]),
                    tenant_id=str(payload.get("tenant_id") or "default"),
                )
                await AgentRunService.mark_persisted(str(payload["run_id"]))
            elif event.event_type == "conversation.checkpoint_delete":
                await delete_thread_checkpoint(
                    int(payload["conversation_id"]),
                    user_id=int(payload["user_id"]),
                    tenant_id=str(payload.get("tenant_id") or "default"),
                )
            elif event.event_type == "memory.index.upsert":
                await MemoryIndexService.upsert_from_event(payload)
            elif event.event_type == "memory.index.delete":
                await MemoryIndexService.delete_from_event(payload)
            else:
                raise ValueError(f"未知 outbox event_type: {event.event_type}")
            await cls._finish(event.id, success=True)
        except Exception as exc:
            logger.error("outbox event {} failed: {}", event.id, exc)
            await cls._finish(event.id, success=False, error=str(exc))
        return True

    @classmethod
    async def run_forever(cls, *, poll_interval: float = 2.0) -> None:
        """持续处理 outbox；通过进程信号停止 worker。"""
        while True:
            processed = await cls.process_once()
            if not processed:
                await asyncio.sleep(poll_interval)
