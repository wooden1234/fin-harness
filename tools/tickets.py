"""人工接管工单工具占位。"""

from __future__ import annotations

from typing import Any


async def create_handoff_ticket(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "not_configured", "payload": payload}
