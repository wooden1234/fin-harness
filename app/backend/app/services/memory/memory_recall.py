"""首期长期记忆精确召回。"""

from __future__ import annotations

from typing import Any

from app.services.memory.memory_service import MemoryService


async def recall_preferences(*, tenant_id: str, user_id: int) -> dict[str, Any]:
    records = await MemoryService.list(tenant_id=tenant_id, user_id=user_id)
    return {
        record.memory_key: (record.value_json or {}).get("value")
        for record in records
    }
