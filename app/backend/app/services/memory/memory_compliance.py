"""管理员合规查询、敏感信息扫描和撤销服务。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.memory.memory_event import MemoryEvent
from app.models.memory.memory_record import MemoryRecord
from app.models.persistence.outbox_event import OutboxEvent

_SENSITIVE_RULES = (
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("asset_or_holding", re.compile(r"持仓|资产|负债|银行卡|账户余额")),
)


def scan_memory(record: MemoryRecord) -> dict[str, Any]:
    text = f"{record.display_text} {record.search_text}"
    matches = [name for name, pattern in _SENSITIVE_RULES if pattern.search(text)]
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
        *, memory_id: str, tenant_id: str, actor_id: int, reason: str
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
                    },
                    status="pending",
                )
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
            return record
