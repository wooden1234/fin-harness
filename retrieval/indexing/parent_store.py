"""PostgreSQL parent chunk store for RAG context recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from retrieval.indexing.index import _pg_connection_strings

RAG_SCHEMA = "rag"


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_parent_store_schema(conn: psycopg.Connection) -> None:
    """创建 RAG 文档、父块和 L3 注册表。"""
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {RAG_SCHEMA}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_SCHEMA}.documents (
                doc_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                file TEXT,
                authority_tier TEXT,
                issuer TEXT,
                effective_date TEXT,
                ticker TEXT,
                fiscal_year INTEGER,
                doc_group TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_SCHEMA}.parent_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES {RAG_SCHEMA}.documents(doc_id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                chunk_level TEXT NOT NULL,
                parent_chunk_id TEXT,
                root_chunk_id TEXT,
                section_path TEXT,
                page_range TEXT,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                child_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                leaf_child_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                text_chars INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RAG_SCHEMA}.chunk_registry (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES {RAG_SCHEMA}.documents(doc_id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                parent_chunk_id TEXT,
                root_chunk_id TEXT,
                chunk_index INTEGER,
                page_num INTEGER,
                section_path TEXT,
                block_type TEXT,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS ix_parent_chunks_doc_level ON {RAG_SCHEMA}.parent_chunks (doc_id, chunk_level)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS ix_parent_chunks_category ON {RAG_SCHEMA}.parent_chunks (category)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS ix_chunk_registry_parent ON {RAG_SCHEMA}.chunk_registry (parent_chunk_id)"
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _document_metadata_from_leaf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = dict((rows[0].get("metadata") or {}) if rows else {})
    return {
        "doc_id": metadata.get("doc_id"),
        "category": metadata.get("category"),
        "title": metadata.get("title"),
        "source": metadata.get("source"),
        "file": metadata.get("file"),
        "authority_tier": metadata.get("authority_tier"),
        "issuer": metadata.get("issuer"),
        "effective_date": metadata.get("effective_date"),
        "ticker": metadata.get("ticker"),
        "fiscal_year": _to_int(metadata.get("fiscal_year")),
        "doc_group": metadata.get("doc_group"),
        "metadata": metadata,
    }


def _parent_links(parent_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    links: dict[str, dict[str, str]] = {}
    for row in parent_rows:
        metadata = row.get("metadata") or {}
        if metadata.get("chunk_level") != "L2":
            continue
        l2_id = str(metadata.get("chunk_id") or "")
        root_id = str(metadata.get("root_chunk_id") or "")
        for child_id in metadata.get("child_chunk_ids", []):
            links[str(child_id)] = {
                "parent_chunk_id": l2_id,
                "root_chunk_id": root_id,
            }
    return links


def upsert_document(conn: psycopg.Connection, doc: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {RAG_SCHEMA}.documents (
                doc_id, category, title, source, file, authority_tier, issuer,
                effective_date, ticker, fiscal_year, doc_group, metadata, updated_at
            ) VALUES (
                %(doc_id)s, %(category)s, %(title)s, %(source)s, %(file)s, %(authority_tier)s,
                %(issuer)s, %(effective_date)s, %(ticker)s, %(fiscal_year)s, %(doc_group)s,
                %(metadata)s, now()
            )
            ON CONFLICT (doc_id) DO UPDATE SET
                category = EXCLUDED.category,
                title = EXCLUDED.title,
                source = EXCLUDED.source,
                file = EXCLUDED.file,
                authority_tier = EXCLUDED.authority_tier,
                issuer = EXCLUDED.issuer,
                effective_date = EXCLUDED.effective_date,
                ticker = EXCLUDED.ticker,
                fiscal_year = EXCLUDED.fiscal_year,
                doc_group = EXCLUDED.doc_group,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            {**doc, "metadata": Jsonb(doc.get("metadata") or {})},
        )


def upsert_parent_chunks(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            cur.execute(
                f"""
                INSERT INTO {RAG_SCHEMA}.parent_chunks (
                    chunk_id, doc_id, category, chunk_level, parent_chunk_id, root_chunk_id,
                    section_path, page_range, text, metadata, child_chunk_ids,
                    leaf_child_chunk_ids, text_chars, updated_at
                ) VALUES (
                    %(chunk_id)s, %(doc_id)s, %(category)s, %(chunk_level)s,
                    %(parent_chunk_id)s, %(root_chunk_id)s, %(section_path)s,
                    %(page_range)s, %(text)s, %(metadata)s, %(child_chunk_ids)s,
                    %(leaf_child_chunk_ids)s, %(text_chars)s, now()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    parent_chunk_id = EXCLUDED.parent_chunk_id,
                    root_chunk_id = EXCLUDED.root_chunk_id,
                    section_path = EXCLUDED.section_path,
                    page_range = EXCLUDED.page_range,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    child_chunk_ids = EXCLUDED.child_chunk_ids,
                    leaf_child_chunk_ids = EXCLUDED.leaf_child_chunk_ids,
                    text_chars = EXCLUDED.text_chars,
                    updated_at = now()
                """,
                {
                    "chunk_id": metadata.get("chunk_id"),
                    "doc_id": metadata.get("doc_id"),
                    "category": metadata.get("category"),
                    "chunk_level": metadata.get("chunk_level"),
                    "parent_chunk_id": metadata.get("parent_chunk_id"),
                    "root_chunk_id": metadata.get("root_chunk_id"),
                    "section_path": metadata.get("section_path"),
                    "page_range": metadata.get("page_range"),
                    "text": row.get("text") or "",
                    "metadata": Jsonb(metadata),
                    "child_chunk_ids": Jsonb(metadata.get("child_chunk_ids") or []),
                    "leaf_child_chunk_ids": Jsonb(metadata.get("leaf_child_chunk_ids") or []),
                    "text_chars": _to_int(metadata.get("text_chars")) or len(row.get("text") or ""),
                },
            )


def upsert_chunk_registry(
    conn: psycopg.Connection,
    leaf_rows: list[dict[str, Any]],
    links: dict[str, dict[str, str]],
) -> None:
    with conn.cursor() as cur:
        for row in leaf_rows:
            metadata = dict(row.get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or "")
            chunk_index = _to_int(metadata.get("chunk_index")) or 0
            chunk_id = str(metadata.get("chunk_id") or f"{doc_id}:L3:{chunk_index:06d}")
            link = links.get(chunk_id) or {}
            metadata["chunk_id"] = chunk_id
            metadata["chunk_level"] = "L3"
            metadata["parent_chunk_id"] = link.get("parent_chunk_id")
            metadata["root_chunk_id"] = link.get("root_chunk_id")
            cur.execute(
                f"""
                INSERT INTO {RAG_SCHEMA}.chunk_registry (
                    chunk_id, doc_id, category, parent_chunk_id, root_chunk_id,
                    chunk_index, page_num, section_path, block_type, metadata, updated_at
                ) VALUES (
                    %(chunk_id)s, %(doc_id)s, %(category)s, %(parent_chunk_id)s,
                    %(root_chunk_id)s, %(chunk_index)s, %(page_num)s,
                    %(section_path)s, %(block_type)s, %(metadata)s, now()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    parent_chunk_id = EXCLUDED.parent_chunk_id,
                    root_chunk_id = EXCLUDED.root_chunk_id,
                    chunk_index = EXCLUDED.chunk_index,
                    page_num = EXCLUDED.page_num,
                    section_path = EXCLUDED.section_path,
                    block_type = EXCLUDED.block_type,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                {
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id"),
                    "category": metadata.get("category"),
                    "parent_chunk_id": metadata.get("parent_chunk_id"),
                    "root_chunk_id": metadata.get("root_chunk_id"),
                    "chunk_index": _to_int(metadata.get("chunk_index")),
                    "page_num": _to_int(metadata.get("page_num")),
                    "section_path": metadata.get("section_path"),
                    "block_type": metadata.get("block_type"),
                    "metadata": Jsonb(metadata),
                },
            )


def index_parent_store(
    *,
    input_dir: Path,
    doc_ids: set[str] | None = None,
    rebuild: bool = False,
) -> dict[str, int]:
    """把 cleaned_v2 的 documents、parent_chunks、chunk_registry 写入 PG。"""
    sync_url, _ = _pg_connection_strings()
    counts = {"documents": 0, "parent_chunks": 0, "leaf_chunks": 0}
    with psycopg.connect(sync_url) as conn:
        ensure_parent_store_schema(conn)
        for chunks_path in sorted(input_dir.glob("**/chunks.jsonl")):
            if doc_ids and chunks_path.parent.name not in doc_ids:
                continue
            parent_path = chunks_path.with_name("parent_chunks.jsonl")
            if not parent_path.exists():
                raise FileNotFoundError(f"缺少 parent chunks: {parent_path}")
            leaf_rows = _load_jsonl(chunks_path)
            parent_rows = _load_jsonl(parent_path)
            if not leaf_rows:
                continue
            doc = _document_metadata_from_leaf(leaf_rows)
            if rebuild:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {RAG_SCHEMA}.documents WHERE doc_id = %s",
                        (doc["doc_id"],),
                    )
            upsert_document(conn, doc)
            links = _parent_links(parent_rows)
            upsert_parent_chunks(conn, parent_rows)
            upsert_chunk_registry(conn, leaf_rows, links)
            counts["documents"] += 1
            counts["parent_chunks"] += len(parent_rows)
            counts["leaf_chunks"] += len(leaf_rows)
        conn.commit()
    return counts
