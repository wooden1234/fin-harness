"""长期记忆 Store 生命周期与索引写入适配。"""

from __future__ import annotations

from contextlib import AsyncExitStack
from time import perf_counter

from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig

from app.core.config import settings
from retrieval.clients.embeddings import get_embed_model
from app.services.memory.memory_catalog import MEMORY_TYPES
from app.services.memory.memory_metrics import increment, observe

_store: AsyncPostgresStore | None = None
_exit_stack: AsyncExitStack | None = None


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    started_at = perf_counter()
    result = await get_embed_model().aget_text_embedding_batch(texts)
    increment("memory_embedding_requests_total")
    increment("memory_embedding_texts_total", len(texts))
    observe("memory_embedding_latency_ms_total", (perf_counter() - started_at) * 1000)
    return result


def memory_namespace(
    tenant_id: str,
    user_id: int,
    memory_type: str,
) -> tuple[str, str, str]:
    return (str(tenant_id), str(user_id), memory_type)


def _require_vectorizable(memory_type: str) -> None:
    definition = MEMORY_TYPES.get(memory_type)
    if definition is None or not definition.vectorizable:
        raise ValueError(f"memory_type_not_vectorizable:{memory_type}")


async def init_memory_store() -> AsyncPostgresStore | None:
    global _store, _exit_stack
    if _store is not None:
        return _store
    if not settings.PGVECTOR_DATABASE_URL:
        return None
    _exit_stack = AsyncExitStack()
    _store = await _exit_stack.enter_async_context(
        AsyncPostgresStore.from_conn_string(
            settings.PGVECTOR_DATABASE_URL,
            index=PostgresIndexConfig(
                dims=settings.EMBEDDING_DIM,
                embed=_embed_texts,
                fields=["search_text"],
            ),
        )
    )
    await _store.setup()
    return _store


def get_memory_store() -> AsyncPostgresStore | None:
    return _store


async def close_memory_store() -> None:
    global _store, _exit_stack
    if _exit_stack is not None:
        await _exit_stack.aclose()
    _store = None
    _exit_stack = None


async def upsert_memory_index(
    *,
    memory_id: str,
    tenant_id: str,
    user_id: int,
    memory_type: str,
    search_text: str,
    version: int,
) -> None:
    _require_vectorizable(memory_type)
    store = get_memory_store()
    if store is None:
        raise RuntimeError("长期记忆 Store 未初始化")
    await store.aput(
        memory_namespace(tenant_id, user_id, memory_type),
        memory_id,
        {
            "memory_id": memory_id,
            "version": version,
            "search_text": search_text,
        },
    )


async def search_memory_ids(
    *,
    tenant_id: str,
    user_id: int,
    memory_type: str,
    query: str,
    limit: int,
) -> list[str]:
    """向量层只暴露候选 ID，业务字段必须返回 PostgreSQL 核验。"""
    _require_vectorizable(memory_type)
    if not query.strip() or limit <= 0:
        return []
    store = get_memory_store()
    if store is None:
        return []
    items = await store.asearch(
        memory_namespace(tenant_id, user_id, memory_type),
        query=query,
        limit=limit,
    )
    return list(dict.fromkeys(str(item.key) for item in items))


async def list_memory_index_entries(
    *,
    tenant_id: str,
    user_id: int,
    memory_type: str,
    page_size: int = 500,
) -> list[tuple[str, int | None]]:
    """供一致性扫描读取索引键和版本；不作为 Agent 召回接口。"""
    _require_vectorizable(memory_type)
    store = get_memory_store()
    if store is None:
        return []
    result: list[tuple[str, int | None]] = []
    offset = 0
    while True:
        items = await store.asearch(
            memory_namespace(tenant_id, user_id, memory_type),
            query=None,
            limit=max(1, min(page_size, 5000)),
            offset=offset,
        )
        for item in items:
            value = item.value if isinstance(item.value, dict) else {}
            version = value.get("version")
            result.append(
                (
                    str(item.key),
                    int(version) if version is not None else None,
                )
            )
        if len(items) < max(1, min(page_size, 5000)):
            break
        offset += len(items)
    return result


async def delete_memory_index(
    *, memory_id: str, tenant_id: str, user_id: int, memory_type: str = "preference"
) -> None:
    store = get_memory_store()
    if store is None:
        raise RuntimeError("长期记忆 Store 未初始化")
    await store.adelete(memory_namespace(tenant_id, user_id, memory_type), memory_id)
