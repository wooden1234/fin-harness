"""审计事件写入。"""

from __future__ import annotations

from audit.models import AuditEvent
from app.core.logger import get_logger

logger = get_logger(service="audit")


def record_audit_event(event: AuditEvent) -> None:
    logger.info(
        "audit event_type={} trace_id={} payload_keys={}",
        event.event_type,
        event.trace_id,
        sorted(event.payload),
    )
