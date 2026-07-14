"""Elasticsearch BM25 索引写入（ingest 双写 + PG 全量重建共用）。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from retrieval.collections import get_collection_registry, get_table_name
from retrieval.es_client import create_es_client, index_name

logger = get_logger(service="es_index")

DEFAULT_TEXT_ANALYZER = "ik_max_word"
DEFAULT_SEARCH_ANALYZER = "ik_smart"
DEFAULT_BATCH_SIZE = 500


def elasticsearch_configured() -> bool:
    return bool(settings.ELASTICSEARCH_URL)


def index_mapping(text_analyzer: str, search_analyzer: str) -> dict[str, Any]:
    text_field = {
        "type": "text",
        "analyzer": text_analyzer,
        "search_analyzer": search_analyzer,
    }
    return {
        "mappings": {
            "dynamic": False,
            "properties": {
                "node_id": {"type": "keyword"},
                "category": {"type": "keyword"},
                "collection": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "file": {"type": "keyword"},
                "title": text_field,
                "section": text_field,
                "company": {"type": "keyword"},
                "ticker": {"type": "keyword"},
                "issuer": {"type": "keyword"},
                "doc_group": {"type": "keyword"},
                "fiscal_year": {"type": "integer"},
                "year": {"type": "integer"},
                "page_num": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "text": text_field,
                "metadata": {"type": "object", "enabled": False},
            },
        },
    }


def ensure_index(
    es: Any,
    name: str,
    *,
    rebuild: bool,
    text_analyzer: str = DEFAULT_TEXT_ANALYZER,
    search_analyzer: str = DEFAULT_SEARCH_ANALYZER,
) -> None:
    existing = es.options(ignore_status=404).indices.get(index=name)
    status = getattr(getattr(existing, "meta", None), "status", None)
    exists = status != 404
    if exists and rebuild:
        es.indices.delete(index=name)
        exists = False
    if not exists:
        es.indices.create(
            index=name,
            body=index_mapping(text_analyzer, search_analyzer),
        )


def coerce_metadata(metadata: Any) -> dict[str, Any]:
    import json

    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_year(metadata: dict[str, Any]) -> int | None:
    effective_date = str(metadata.get("effective_date") or "")
    if len(effective_date) >= 4:
        return to_int(effective_date[:4])
    return None


def build_document(
    *,
    category: str,
    collection: str,
    node_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(metadata)
    year = to_int(metadata.get("fiscal_year")) or infer_year(metadata)
    return {
        "node_id": node_id,
        "category": category,
        "collection": collection,
        "doc_id": metadata.get("doc_id"),
        "source": metadata.get("source"),
        "file": metadata.get("file") or metadata.get("file_name"),
        "title": metadata.get("title"),
        "section": metadata.get("section"),
        "company": metadata.get("company"),
        "ticker": metadata.get("ticker"),
        "issuer": metadata.get("issuer"),
        "doc_group": metadata.get("doc_group"),
        "fiscal_year": to_int(metadata.get("fiscal_year")),
        "year": year,
        "page_num": to_int(metadata.get("page_num")),
        "chunk_index": to_int(metadata.get("chunk_index")),
        "text": text,
        "metadata": metadata,
    }


def documents_from_nodes(
    category: str,
    nodes: Iterable[Any],
) -> list[dict[str, Any]]:
    collection = get_table_name(category)
    documents: list[dict[str, Any]] = []
    for node in nodes:
        metadata = dict(getattr(node, "metadata", None) or {})
        ref_doc_id = getattr(node, "ref_doc_id", None)
        if ref_doc_id and not metadata.get("doc_id"):
            metadata["doc_id"] = ref_doc_id
        text = (
            node.get_content(metadata_mode="none")
            if hasattr(node, "get_content")
            else str(getattr(node, "text", "") or "")
        )
        documents.append(
            build_document(
                category=category,
                collection=collection,
                node_id=str(node.node_id),
                text=text,
                metadata=metadata,
            )
        )
    return documents


def bulk_index_documents(
    documents: Iterable[dict[str, Any]],
    *,
    category: str,
    rebuild: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    text_analyzer: str = DEFAULT_TEXT_ANALYZER,
    search_analyzer: str = DEFAULT_SEARCH_ANALYZER,
    es: Any | None = None,
) -> tuple[int, int]:
    """Bulk index prepared ES docs. Returns (success_count, error_count)."""
    from elasticsearch.helpers import bulk

    client = es or create_es_client()
    name = index_name(category)
    ensure_index(
        client,
        name,
        rebuild=rebuild,
        text_analyzer=text_analyzer,
        search_analyzer=search_analyzer,
    )

    def actions() -> Iterator[dict[str, Any]]:
        for doc in documents:
            node_id = str(doc.get("node_id") or "")
            yield {
                "_op_type": "index",
                "_index": name,
                "_id": f"{category}:{node_id}",
                "_source": doc,
            }

    success, errors = bulk(
        client,
        actions(),
        chunk_size=batch_size,
        raise_on_error=False,
        stats_only=False,
    )
    error_count = len(errors) if isinstance(errors, list) else int(errors or 0)
    return int(success), error_count


def index_nodes_to_elasticsearch(
    category: str,
    nodes: Iterable[Any],
    *,
    rebuild: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    require_configured: bool = False,
) -> int:
    """
    将 LlamaIndex nodes 写入 ES BM25 索引。

    ELASTICSEARCH_URL 未配置时默认跳过；require_configured=True 则抛错。
    """
    if not elasticsearch_configured():
        if require_configured:
            raise RuntimeError("未配置 ELASTICSEARCH_URL，无法写入 Elasticsearch")
        logger.info(
            "skip es dual-write category={} reason=ELASTICSEARCH_URL_empty",
            category,
        )
        return 0

    documents = documents_from_nodes(category, nodes)
    if not documents and not rebuild:
        return 0

    success, error_count = bulk_index_documents(
        documents,
        category=category,
        rebuild=rebuild,
        batch_size=batch_size,
    )
    if error_count:
        raise RuntimeError(
            f"Elasticsearch bulk index failed category={category} "
            f"indexed={success} errors={error_count}"
        )
    logger.info(
        "es dual-write category={} indexed={} rebuild={} index={}",
        category,
        success,
        rebuild,
        index_name(category),
    )
    return success


def selected_collections(categories: list[str] | None) -> dict[str, str]:
    registry = get_collection_registry()
    if not categories:
        return registry
    unknown = [category for category in categories if category not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ValueError(f"未知 categories={unknown}，可选: {known}")
    return {category: registry[category] for category in categories}
