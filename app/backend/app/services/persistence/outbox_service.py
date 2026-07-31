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
from app.services.memory.memory_cache import MemoryCache
from app.services.memory.memory_audit import MemoryAuditContext
from app.services.memory.memory_command import parse_memory_rule_action
from app.services.memory.memory_extraction import extract_preference
from app.services.memory.memory_service import MemoryService
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
    async def enqueue_memory_extraction(
        *,
        run_id: str,
        user_id: int,
        source_text: str,
        tenant_id: str = "default",
        conversation_id: int | None = None,
        message_id: int | None = None,
        agent_id: str | None = None,
        task_id: str | None = None,
        trace_id: str | None = None,
    ) -> OutboxEvent:
        """登记回答后执行的隐含偏好提取任务。"""
        event_key = f"memory:extract:{run_id}"
        payload = {
            "run_id": run_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "source_text": source_text,
            "agent_id": agent_id,
            "task_id": task_id,
            "trace_id": trace_id or run_id,
        }
        async with AsyncSessionLocal() as db:
            existing = await db.scalar(
                select(OutboxEvent).where(OutboxEvent.event_key == event_key)
            )
            if existing is not None:
                return existing
            event = OutboxEvent(
                event_key=event_key,
                event_type="memory.preference.extract",
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
            elif event.event_type == "memory.cache.invalidate":
                await MemoryCache.invalidate_user(
                    tenant_id=str(payload["tenant_id"]),
                    user_id=int(payload["user_id"]),
                    memory_type=str(payload.get("memory_type") or "preference"),
                    raise_on_error=True,
                )
            elif event.event_type == "memory.preference.extract":
                source_text = str(payload["source_text"])
                # 防御性复核：同步、临时和敏感动作绝不进入大模型提取。
                if parse_memory_rule_action(source_text).kind == "implicit":
                    preference = await extract_preference(source_text)
                    if preference is not None:
                        await MemoryService.create(
                            tenant_id=str(payload.get("tenant_id") or "default"),
                            user_id=int(payload["user_id"]),
                            memory_key=preference.memory_key,
                            value=preference.value,
                            provenance={
                                "source_type": preference.source,
                                "conversation_id": payload.get("conversation_id"),
                                "message_id": payload.get("message_id"),
                                "run_id": payload.get("run_id"),
                                "evidence": preference.evidence[:500],
                                "excerpt": source_text[:500],
                            },
                            actor_id=str(payload["user_id"]),
                            confidence=preference.confidence,
                            audit_context=MemoryAuditContext(
                                tenant_id=str(
                                    payload.get("tenant_id") or "default"
                                ),
                                user_id=int(payload["user_id"]),
                                agent_id=payload.get("agent_id"),
                                task_id=payload.get("task_id"),
                                trace_id=payload.get("trace_id"),
                            ),
                        )
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
