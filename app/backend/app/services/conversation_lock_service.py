"""跨进程会话租约锁。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.database import AsyncSessionLocal
from app.models.conversation_lock import ConversationLock


class ConversationBusyError(RuntimeError):
    """会话已有其他运行占用。"""


class ConversationLockService:
    @staticmethod
    async def acquire(
        conversation_id: int, *, lease_seconds: int = 1800
    ) -> str:
        token = str(uuid4())
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            try:
                lock = await db.scalar(
                    select(ConversationLock)
                    .where(ConversationLock.conversation_id == conversation_id)
                    .with_for_update()
                )
                if lock is None:
                    db.add(
                        ConversationLock(
                            conversation_id=conversation_id,
                            lock_token=token,
                            expires_at=now + timedelta(seconds=lease_seconds),
                        )
                    )
                elif lock.expires_at > now:
                    raise ConversationBusyError("会话已有进行中的 Agent 运行")
                else:
                    lock.lock_token = token
                    lock.acquired_at = now
                    lock.expires_at = now + timedelta(seconds=lease_seconds)
                await db.commit()
                return token
            except IntegrityError as exc:
                await db.rollback()
                raise ConversationBusyError("会话已有进行中的 Agent 运行") from exc

    @staticmethod
    async def release(conversation_id: int, token: str) -> None:
        async with AsyncSessionLocal() as db:
            lock = await db.scalar(
                select(ConversationLock).where(
                    ConversationLock.conversation_id == conversation_id,
                    ConversationLock.lock_token == token,
                )
            )
            if lock is not None:
                await db.delete(lock)
                await db.commit()

