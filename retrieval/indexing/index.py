from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent.parent
_BACKEND_DIR = _ROOT_DIR / "app" / "backend"
if sys.path and os.path.abspath(sys.path[0]) == str(_SCRIPT_DIR):
    sys.path.pop(0)
for _path in (str(_BACKEND_DIR), str(_ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(_ROOT_DIR / ".env")

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from retrieval.clients.embeddings import get_embed_model
from retrieval.core.collections import get_collection_registry, get_table_name
from retrieval.indexing.es_index import index_nodes_to_elasticsearch

EMBED_DIM = settings.EMBEDDING_DIM
VECTOR_SCHEMA = "rag"

# 兼容旧代码：FAQ 集合表名
TABLE_NAME = get_table_name("faq")


def _pg_connection_strings() -> tuple[str, str]:
    """同步 / 异步连接串，供 PGVectorStore 使用。"""
    sync_url = settings.PGVECTOR_DATABASE_URL
    if not sync_url:
        raw = settings.DATABASE_URL
        sync_url = raw.replace("postgresql+asyncpg://", "postgresql://")
    if not sync_url:
        raise RuntimeError(
            "未配置 PGVECTOR_DATABASE_URL 或 DATABASE_URL，无法连接向量库"
        )
    async_url = sync_url.replace("postgresql://", "postgresql+asyncpg://")
    return sync_url, async_url


def _pg_table_name(table_name: str) -> str:
    """LlamaIndex PGVectorStore 实际表名为 data_{table_name}。"""
    return f"data_{table_name}"


def _drop_vector_table(table_name: str) -> None:
    """删除 pgvector 表（含索引），以便按新 embed_dim 重建 schema。"""
    sync_url, _ = _pg_connection_strings()
    engine = create_engine(sync_url)
    pg_table = _pg_table_name(table_name)
    with engine.begin() as conn:
        conn.execute(
            text(f'DROP TABLE IF EXISTS "{VECTOR_SCHEMA}"."{pg_table}" CASCADE')
        )


def _vector_table_embed_dim(table_name: str) -> int | None:
    """读取现有 embedding 列维度；表不存在时返回 None。"""
    sync_url, _ = _pg_connection_strings()
    engine = create_engine(sync_url)
    pg_table = _pg_table_name(table_name)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = :schema
                  AND c.relname = :table
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            ),
            {"schema": VECTOR_SCHEMA, "table": pg_table},
        ).fetchone()
    return int(row[0]) if row else None


def get_vector_store(
    table_name: str | None = None,
    *,
    category: str = "faq",
    rebuild: bool = False,
) -> PGVectorStore:
    """按 category 或显式 table_name 获取 PGVectorStore。"""
    resolved = table_name or get_table_name(category)
    sync_url, async_url = _pg_connection_strings()
    existing_dim = _vector_table_embed_dim(resolved)
    if rebuild or (existing_dim is not None and existing_dim != EMBED_DIM):
        if existing_dim is not None and existing_dim != EMBED_DIM:
            print(
                f"pgvector dim mismatch: table={_pg_table_name(resolved)} "
                f"has {existing_dim}, EMBEDDING_DIM={EMBED_DIM}; dropping table"
            )
        _drop_vector_table(resolved)
    vector_store = PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=resolved,
        embed_dim=EMBED_DIM,
        schema_name=VECTOR_SCHEMA,
        perform_setup=True,
    )
    if rebuild and existing_dim == EMBED_DIM:
        vector_store.clear()
    return vector_store


def build_index(
    nodes: list[TextNode],
    *,
    category: str = "faq",
    table_name: str | None = None,
    rebuild: bool = False,
    show_progress: bool = True,
    sync_elasticsearch: bool = True,
) -> VectorStoreIndex:
    """将 ingest 得到的 nodes 写入指定 pgvector 集合，并可双写 ES BM25。"""
    embed_model = get_embed_model()
    Settings.embed_model = embed_model
    resolved = table_name or get_table_name(category)
    vector_store = get_vector_store(resolved, rebuild=rebuild)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=show_progress,
    )
    if sync_elasticsearch:
        indexed = index_nodes_to_elasticsearch(
            category,
            nodes,
            rebuild=rebuild,
        )
        if indexed:
            print(f"es indexed: category={category} nodes={indexed} rebuild={rebuild}")
    return index


@lru_cache(maxsize=None)
def load_index(category: str = "faq") -> VectorStoreIndex:
    """从已有 PG 表加载索引（检索用，不重新 embed）。"""
    embed_model = get_embed_model()
    Settings.embed_model = embed_model
    vector_store = get_vector_store(category=category, rebuild=False)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )


def build_indexes_by_category(
    nodes_by_category: dict[str, list[TextNode]],
    *,
    rebuild: bool = True,
    show_progress: bool = True,
    sync_elasticsearch: bool = True,
) -> dict[str, int]:
    """按 category 分组写入各自 pgvector 集合，并可双写 ES。"""
    counts: dict[str, int] = {}
    for category, nodes in nodes_by_category.items():
        if not nodes:
            continue
        build_index(
            nodes,
            category=category,
            rebuild=rebuild,
            show_progress=show_progress,
            sync_elasticsearch=sync_elasticsearch,
        )
        counts[category] = len(nodes)
    return counts


def main() -> None:
    import argparse

    from retrieval.indexing.ingest import run_ingest

    parser = argparse.ArgumentParser(description="构建 FAQ 向量索引")
    parser.add_argument("--rebuild", action="store_true", help="清空表后全量重建")
    args = parser.parse_args()
    nodes = run_ingest()
    print(f"nodes: {len(nodes)}")
    build_index(nodes, category="faq", rebuild=args.rebuild)
    print(f"index built → table={get_table_name('faq')}, dim={EMBED_DIM} (pg + es if configured)")
    print("collections:", get_collection_registry())


if __name__ == "__main__":
    main()
