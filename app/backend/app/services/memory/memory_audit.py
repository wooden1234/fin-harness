"""长期记忆审计上下文与审计写入辅助。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.logger import get_logger
from app.models.persistence.audit_log import AuditLog

logger = get_logger(service="memory_audit")


@dataclass(frozen=True, slots=True)
class MemoryAuditContext:
    tenant_id: str
    user_id: int
    agent_id: str | None = None
    task_id: str | None = None
    trace_id: str | None = None


def context_for(
    *,
    tenant_id: str,
    user_id: int,
    audit_context: MemoryAuditContext | None = None,
) -> MemoryAuditContext:
    if audit_context is not None:
        if (
            str(audit_context.tenant_id) != str(tenant_id)
            or int(audit_context.user_id) != int(user_id)
        ):
            raise ValueError("memory_audit_scope_mismatch")
        return audit_context
    return MemoryAuditContext(tenant_id=str(tenant_id), user_id=int(user_id))


def add_audit(
    db: AsyncSession,
    *,
    context: MemoryAuditContext,
    action: str,
    resource_id: str | None = None,
    memory_type: str | None = None,
    memory_key: str | None = None,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """把审计行加入当前业务事务，确保业务与审计同提交。"""
    db.add(
        AuditLog(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            action=action,
            resource_type="memory",
            resource_id=resource_id,
            memory_type=memory_type,
            memory_key=memory_key,
            agent_id=context.agent_id,
            task_id=context.task_id,
            trace_id=context.trace_id,
            details=details or {},
            error_message=error_message,
        )
    )


async def record_audit(
    *,
    context: MemoryAuditContext,
    action: str,
    resource_id: str | None = None,
    memory_type: str | None = None,
    memory_key: str | None = None,
    details: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """为读操作或异步边界单独写审计；审计故障不阻断记忆主流程。"""
    try:
        async with AsyncSessionLocal() as db:
            add_audit(
                db,
                context=context,
                action=action,
                resource_id=resource_id,
                memory_type=memory_type,
                memory_key=memory_key,
                details=details,
                error_message=error_message,
            )
            await db.commit()
    except Exception as exc:
        logger.warning("记忆审计写入失败: {}", type(exc).__name__)


__all__ = ["MemoryAuditContext", "add_audit", "context_for", "record_audit"]
