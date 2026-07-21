"""用户数据导出与删除审计服务。"""

from __future__ import annotations

from sqlalchemy import select, update, func

from app.core.database import AsyncSessionLocal
from app.models.audit_log import AuditLog
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.agent_run import AgentRun
from app.models.outbox_event import OutboxEvent


class PrivacyService:
    @staticmethod
    async def export_user_data(user_id: int) -> dict:
        async with AsyncSessionLocal() as db:
            conversations = (await db.execute(
                select(Conversation).where(Conversation.user_id == user_id)
            )).scalars().all()
            conversation_ids = [row.id for row in conversations]
            messages = []
            runs = []
            if conversation_ids:
                messages = (await db.execute(
                    select(Message).where(Message.conversation_id.in_(conversation_ids))
                )).scalars().all()
                runs = (await db.execute(
                    select(AgentRun).where(AgentRun.conversation_id.in_(conversation_ids))
                )).scalars().all()
            db.add(AuditLog(
                user_id=user_id, action="user.data_export", resource_type="user",
                resource_id=str(user_id), details={"conversation_count": len(conversations)},
            ))
            await db.commit()
            return {
                "user_id": user_id,
                "conversations": [
                    {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat() if c.created_at else None,
                     "deleted_at": c.deleted_at.isoformat() if c.deleted_at else None}
                    for c in conversations
                ],
                "messages": [
                    {"id": m.id, "conversation_id": m.conversation_id, "sender": m.sender,
                     "content": m.content, "run_id": m.run_id, "sequence_no": m.sequence_no,
                     "created_at": m.created_at.isoformat() if m.created_at else None}
                    for m in messages
                ],
                "runs": [
                    {"id": r.id, "conversation_id": r.conversation_id, "status": r.status,
                     "thread_id": r.thread_id, "checkpoint_id": r.checkpoint_id,
                     "summary_snapshot": r.summary_snapshot}
                    for r in runs
                ],
            }

    @staticmethod
    async def request_user_data_delete(user_id: int) -> int:
        async with AsyncSessionLocal() as db:
            conversations = (await db.execute(
                select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
            )).scalars().all()
            for conversation in conversations:
                conversation.deleted_at = func.now()
                conversation.deleted_by = user_id
                conversation.status = "deleted"
                db.add(OutboxEvent(
                    event_key=f"conversation_checkpoint_delete:{conversation.id}",
                    event_type="conversation.checkpoint_delete",
                    aggregate_id=str(conversation.id),
                    payload={"conversation_id": conversation.id, "user_id": user_id},
                    status="pending",
                ))
            db.add(AuditLog(
                user_id=user_id, action="user.data_delete_requested", resource_type="user",
                resource_id=str(user_id), details={"conversation_count": len(conversations)},
            ))
            await db.commit()
            return len(conversations)

