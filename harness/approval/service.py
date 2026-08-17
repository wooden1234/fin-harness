"""审批：at-most-once。"""

from __future__ import annotations

from typing import Any, Sequence

from harness.contracts.errors import ApprovalError
from harness.session.types import EventDraft, SessionEvent


def pending_approvals(events: Sequence[SessionEvent]) -> list[dict[str, Any]]:
    asked: dict[str, dict[str, Any]] = {}
    decided: set[str] = set()
    for event in events:
        if event.event_type == "approval/asked":
            approval_id = str(event.data.get("approval_id") or "")
            asked[approval_id] = dict(event.data)
        elif event.event_type == "approval/decided":
            decided.add(str(event.data.get("approval_id") or ""))
    return [item for key, item in asked.items() if key and key not in decided]


def validate_decision(events: Sequence[SessionEvent], approval_id: str) -> dict[str, Any]:
    pending = {item.get("approval_id"): item for item in pending_approvals(events)}
    item = pending.get(approval_id)
    if item is None:
        raise ApprovalError("approval_not_pending", code="not_pending")
    return item
