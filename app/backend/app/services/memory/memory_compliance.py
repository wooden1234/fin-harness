"""管理员合规查询、敏感信息扫描和撤销服务。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.memory.memory_event import MemoryEvent
from app.models.memory.memory_record import MemoryRecord
from app.models.persistence.outbox_event import OutboxEvent
from app.services.memory.memory_audit import MemoryAuditContext, add_audit, context_for
from app.services.memory.memory_cache import MemoryCache
from app.services.memory.memory_policy import find_sensitive_memory_rules


def scan_memory(record: MemoryRecord) -> dict[str, Any]:
    text = f"{record.display_text} {record.search_text}"
    matches = list(find_sensitive_memory_rules(text))
    return {
        "memory_id": record.id,
        "risk": "high" if matches else "none",
        "matched_rules": matches,
        "redacted_preview": "[已脱敏]" if matches else record.display_text,
    }


class MemoryComplianceService:
    @staticmethod
    async def list_records(
        *, tenant_id: str | None = None, user_id: int | None = None, limit: int = 100
    ) -> list[MemoryRecord]:
        async with AsyncSessionLocal() as db:
            statement = select(MemoryRecord)
            if tenant_id is not None:
                statement = statement.where(MemoryRecord.tenant_id == tenant_id)
            if user_id is not None:
                statement = statement.where(MemoryRecord.user_id == user_id)
            result = await db.execute(
                statement.order_by(MemoryRecord.updated_at.desc()).limit(limit)
            )
            return list(result.scalars().all())

    @staticmethod
    async def revoke(
        *,
        memory_id: str,
        tenant_id: str,
        actor_id: int,
        reason: str,
        audit_context: MemoryAuditContext | None = None,
    ) -> MemoryRecord | None:
        async with AsyncSessionLocal() as db:
            record = await db.scalar(
                select(MemoryRecord).where(
                    MemoryRecord.id == memory_id,
                    MemoryRecord.tenant_id == tenant_id,
                    MemoryRecord.status.in_(["active", "pending"]),
                )
            )
            if record is None:
                return None
            context = (
                MemoryAuditContext(
                    tenant_id=tenant_id,
                    user_id=int(record.user_id),
                    agent_id=audit_context.agent_id,
                    task_id=audit_context.task_id,
                    trace_id=audit_context.trace_id,
                )
                if audit_context is not None
                else context_for(
                    tenant_id=tenant_id,
                    user_id=int(record.user_id),
                )
            )
            now = datetime.now(timezone.utc)
            record.status = "revoked"
            record.consent_status = "withdrawn"
            record.withdrawn_at = now
            record.updated_at = now
            db.add(
                OutboxEvent(
                    event_key=f"memory:index:delete:{record.id}:admin:{record.version}",
                    event_type="memory.index.delete",
                    aggregate_id=record.id,
                    payload={
                        "memory_id": record.id,
                        "tenant_id": record.tenant_id,
                        "user_id": record.user_id,
                        "memory_type": record.memory_type,
                        "agent_id": context.agent_id,
                        "task_id": context.task_id,
                        "trace_id": context.trace_id,
                    },
                    status="pending",
                )
            )
            if record.memory_type == "preference":
                db.add(
                    OutboxEvent(
                        event_key=(
                            f"memory:cache:invalidate:{tenant_id}:"
                            f"{record.user_id}:admin:{record.id}:v:{record.version}"
                        ),
                        event_type="memory.cache.invalidate",
                        aggregate_id=f"{tenant_id}:{record.user_id}",
                        payload={
                            "tenant_id": tenant_id,
                            "user_id": int(record.user_id),
                            "memory_type": "preference",
                            "agent_id": context.agent_id,
                            "task_id": context.task_id,
                            "trace_id": context.trace_id,
                        },
                        status="pending",
                    )
                )
            add_audit(
                db,
                context=MemoryAuditContext(
                    tenant_id=tenant_id,
                    user_id=int(record.user_id),
                    agent_id=context.agent_id,
                    task_id=context.task_id,
                    trace_id=context.trace_id,
                ),
                action="memory.delete",
                resource_id=record.id,
                memory_type=record.memory_type,
                memory_key=record.memory_key,
                details={"reason": reason, "admin": True},
            )
            db.add(
                MemoryEvent(
                    id=__import__("uuid").uuid4().hex,
                    tenant_id=record.tenant_id,
                    user_id=record.user_id,
                    memory_id=record.id,
                    event_type="admin_revoked",
                    event_key=f"memory:{record.id}:admin_revoked:{now.timestamp()}",
                    payload_json={"reason": reason},
                    actor_type="admin",
                    actor_id=str(actor_id),
                )
            )
            await db.commit()
            await db.refresh(record)
        if record.memory_type == "preference":
            await MemoryCache.invalidate_user(
                tenant_id=record.tenant_id,
                user_id=int(record.user_id),
            )
        return record
