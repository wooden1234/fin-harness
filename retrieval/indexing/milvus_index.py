"""Milvus L3 leaf chunk indexing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.core.config import settings
from retrieval.clients.milvus_client import collection_name, create_milvus_client
from retrieval.clients.embeddings import get_embed_model, embedding_batch_size

# Milvus 写入批次跟随 embedding batch 限制（DashScope 上限 25）。
DEFAULT_BATCH_SIZE = 25


def _milvus_batch_size() -> int:
    return min(DEFAULT_BATCH_SIZE, embedding_batch_size())


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_collection(client: Any, category: str, *, rebuild: bool) -> str:
    from pymilvus import DataType

    name = collection_name(category)
    if client.has_collection(name):
        if rebuild:
            client.drop_collection(name)
        else:
            return name

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=160)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.MILVUS_DIM)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=96)
    schema.add_field("ticker", DataType.VARCHAR, max_length=32)
    schema.add_field("issuer", DataType.VARCHAR, max_length=256)
    schema.add_field("fiscal_year", DataType.INT64)
    schema.add_field("year", DataType.INT64)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("section_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("block_type", DataType.VARCHAR, max_length=64)
    schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=160)
    schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=160)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("page_num", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)

    index_params = client.prepare_index_params()
    params = {}
    if settings.MILVUS_INDEX_TYPE.upper() == "HNSW":
        params = {
            "M": settings.MILVUS_M,
            "efConstruction": settings.MILVUS_EF_CONSTRUCTION,
        }
    index_params.add_index(
        field_name="embedding",
        index_type=settings.MILVUS_INDEX_TYPE,
        metric_type=settings.MILVUS_METRIC_TYPE,
        params=params,
    )
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    return name


def _chunk_record(chunk: Any, embedding: list[float]) -> dict[str, Any]:
    metadata = dict(getattr(chunk, "metadata", None) or {})
    doc_id = str(metadata.get("doc_id") or "")
    chunk_index = _to_int(metadata.get("chunk_index")) or 0
    chunk_id = str(metadata.get("chunk_id") or f"{doc_id}:L3:{chunk_index:06d}")
    text = str(getattr(chunk, "text", "") or "")
    fiscal_year = _to_int(metadata.get("fiscal_year"))
    effective_date = str(metadata.get("effective_date") or "")
    year = fiscal_year
    if year is None and len(effective_date) >= 4:
        year = _to_int(effective_date[:4])
    return {
        "chunk_id": chunk_id,
        "embedding": embedding,
        "doc_id": doc_id,
        "ticker": str(metadata.get("ticker") or ""),
        "issuer": str(metadata.get("issuer") or ""),
        "fiscal_year": fiscal_year or 0,
        "year": year or 0,
        "category": str(metadata.get("category") or ""),
        "source": str(metadata.get("source") or ""),
        "section_path": str(metadata.get("section_path") or ""),
        "block_type": str(metadata.get("block_type") or ""),
        "parent_chunk_id": str(metadata.get("parent_chunk_id") or ""),
        "root_chunk_id": str(metadata.get("root_chunk_id") or ""),
        "chunk_index": chunk_index,
        "page_num": _to_int(metadata.get("page_num")) or 0,
        "text": text[:8192],
        "metadata": metadata,
    }


def index_chunks_to_milvus(
    chunks_by_category: dict[str, list[Any]],
    *,
    rebuild: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """将 L3 chunks 按 category 写入 Milvus。"""
    client = create_milvus_client()
    embed_model = get_embed_model()
    batch_size = min(max(int(batch_size), 1), _milvus_batch_size())
    counts: dict[str, int] = {}
    for category, chunks in chunks_by_category.items():
        if not chunks:
            continue
        collection = _ensure_collection(client, category, rebuild=rebuild)
        indexed = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [str(getattr(chunk, "text", "") or "") for chunk in batch]
            embeddings = embed_model.get_text_embedding_batch(texts)
            rows = [
                _chunk_record(chunk, embedding)
                for chunk, embedding in zip(batch, embeddings, strict=False)
            ]
            if rows:
                client.insert(collection_name=collection, data=rows)
                indexed += len(rows)
        counts[category] = indexed
    return counts


def group_chunks_by_category(chunks: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        category = str(metadata.get("category") or metadata.get("doc_type") or "")
        if category:
            grouped[category].append(chunk)
    return dict(grouped)
