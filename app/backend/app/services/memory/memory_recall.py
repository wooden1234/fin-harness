"""首期长期记忆精确召回。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from time import perf_counter

from app.services.memory.memory_service import MemoryService
from app.core.config import settings
from app.services.memory.memory_store import get_memory_store, memory_namespace
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
    store = get_memory_store()
    records_by_id = {
        record.id: record
        for record in await MemoryService.list(tenant_id=tenant_id, user_id=user_id)
    }
    selected_ids: list[str] = []
    if store is not None and query.strip():
        try:
            items = await store.asearch(
                memory_namespace(tenant_id, user_id, "preference"),
                query=query,
                limit=top_k,
            )
            selected_ids = [str(item.key) for item in items]
            increment("memory_recall_store_search_total")
        except Exception:
            increment("memory_recall_store_error_total")
    if not selected_ids:
        selected_ids = list(records_by_id)[:top_k]
        increment("memory_recall_sql_fallback_total")

    result: dict[str, Any] = {}
    used_tokens = 0
    budget_skipped = 0
    stale_skipped = 0
    for memory_id in selected_ids:
        record = records_by_id.get(memory_id)
        if record is None:
            stale_skipped += 1
            continue
        # Store 只是候选索引，最终状态必须以 SQL 权威记录二次校验为准。
        if (
            record.tenant_id != tenant_id
            or record.user_id != user_id
            or record.status != "active"
            or (
                record.expires_at is not None
                and record.expires_at <= datetime.now(timezone.utc)
            )
        ):
            stale_skipped += 1
            continue
        cost = _token_cost(record.search_text)
        if used_tokens + cost > token_budget:
            budget_skipped += 1
            continue
        result[record.memory_key] = (record.value_json or {}).get("value")
        used_tokens += cost
    increment("memory_recall_requests_total")
    increment("memory_recall_candidates_total", len(selected_ids))
    increment("memory_recall_hits_total", len(result))
    increment("memory_recall_budget_skipped_total", budget_skipped)
    increment("memory_recall_stale_skipped_total", stale_skipped)
    observe("memory_recall_tokens_total", used_tokens)
    observe("memory_recall_latency_ms_total", (perf_counter() - started_at) * 1000)
    return result
