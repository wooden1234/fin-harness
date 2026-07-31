"""兼容入口：结构化偏好仅通过 PostgreSQL 精确召回。"""

from __future__ import annotations

from typing import Any
from time import perf_counter

from app.services.memory.memory_service import MemoryService
from app.core.config import settings
from app.services.memory.memory_metrics import increment, observe


def _token_cost(text: str) -> int:
    return max(1, len(text) // 4)


async def recall_preferences(
    *,
    tenant_id: str,
    user_id: int,
    query: str = "",
    top_k: int | None = None,
    token_budget: int | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    top_k = top_k or settings.MEMORY_RECALL_TOP_K
    token_budget = token_budget or settings.MEMORY_RECALL_TOKEN_BUDGET
    del query
    records = (
        await MemoryService.list(tenant_id=tenant_id, user_id=user_id)
    )[:top_k]

    result: dict[str, Any] = {}
    used_tokens = 0
    budget_skipped = 0
    for record in records:
        cost = _token_cost(record.search_text)
        if used_tokens + cost > token_budget:
            budget_skipped += 1
            continue
        result[record.memory_key] = (record.value_json or {}).get("value")
        used_tokens += cost
    increment("memory_recall_requests_total")
    increment("memory_recall_candidates_total", len(records))
    increment("memory_recall_hits_total", len(result))
    increment("memory_recall_budget_skipped_total", budget_skipped)
    increment("memory_recall_sql_exact_total")
    observe("memory_recall_tokens_total", used_tokens)
    observe("memory_recall_latency_ms_total", (perf_counter() - started_at) * 1000)
    return result
