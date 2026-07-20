"""Elasticsearch BM25 词法检索。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.config import settings
from retrieval.core.collections import get_collection_registry, get_table_name
from retrieval.clients.es_client import create_es_client, index_name
from retrieval.core.filters import (
    MetadataFilters,
    filter_categories,
    filters_for_category,
    merge_filters,
)

if TYPE_CHECKING:
    from retrieval.retrievers.retriever import RetrievalHit

_ROOT_META_FIELDS = (
    "doc_id",
    "source",
    "file",
    "title",
    "section",
    "company",
    "ticker",
    "issuer",
    "doc_group",
    "fiscal_year",
    "year",
    "page_num",
    "chunk_index",
)


class ElasticsearchBM25Retriever:
    """基于预构建 chunk 索引执行 Elasticsearch BM25 召回。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        metadata_filters: MetadataFilters | None = None,
    ):
        registry = get_collection_registry()
        if categories is None:
            self.categories = list(registry.keys())
        else:
            unknown = [category for category in categories if category not in registry]
            if unknown:
                known = ", ".join(sorted(registry))
                raise ValueError(f"未知 categories={unknown}，可选: {known}")
            self.categories = list(categories)

        self.metadata_filters = metadata_filters or {}
        self._client = create_es_client()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalHit]:
        from retrieval.retrievers.retriever import RetrievalHit

        filters = merge_filters(self.metadata_filters, metadata_filters)
        categories = _filtered_categories(self.categories, filters)
        if not categories:
            return []

        hits: list[RetrievalHit] = []
        for category in categories:
            category_filters = filters_for_category(filters, category)
            response = self._client.search(
                index=[index_name(category)],
                body={
                    "query": {
                        "bool": {
                            "must": [_text_query(query)],
                            "filter": _filter_clauses(category_filters),
                        }
                    },
                    "size": top_k,
                },
            )
            for raw_hit in response.get("hits", {}).get("hits", []):
                source = raw_hit.get("_source") or {}
                metadata = _hit_metadata(source)
                hit_category = str(source.get("category") or metadata.get("category") or category)
                collection = str(
                    source.get("collection")
                    or metadata.get("collection")
                    or (get_table_name(hit_category) if hit_category else "")
                )
                metadata.setdefault("category", hit_category)
                metadata.setdefault("collection", collection)
                metadata["bm25_score"] = float(raw_hit.get("_score") or 0.0)
                metadata["es_index"] = raw_hit.get("_index")
                hits.append(
                    RetrievalHit(
                        text=str(source.get("leaf_text") or source.get("text") or ""),
                        score=float(raw_hit.get("_score") or 0.0),
                        metadata=metadata,
                        node_id=str(source.get("node_id") or "") or None,
                        category=hit_category or None,
                        collection=collection or None,
                        score_type="bm25",
                    )
                )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


def _hit_metadata(source: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(source.get("metadata") or {})
    # 根字段是建索引时提升出来的权威值（含 doc_id）
    for key in _ROOT_META_FIELDS:
        value = source.get(key)
        if value is not None and value != "":
            metadata[key] = value
    return metadata


def _text_query(query: str) -> dict[str, Any]:
    fields = [
        f"leaf_text^{settings.ES_BM25F_LEAF_TEXT_WEIGHT:g}",
        f"title^{settings.ES_BM25F_TITLE_WEIGHT:g}",
        f"section^{settings.ES_BM25F_SECTION_WEIGHT:g}",
        f"source^{settings.ES_BM25F_SOURCE_WEIGHT:g}",
    ]
    if settings.ES_BM25_MODE.strip().lower() == "combined_fields":
        base_query: dict[str, Any] = {
            "combined_fields": {
                "query": query,
                "fields": fields,
                "operator": "or",
            }
        }
    else:
        base_query = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "most_fields",
                "operator": "or",
            }
        }

    return {
        "bool": {
            "must": [base_query],
        }
    }


def _filter_clauses(filters: MetadataFilters | None) -> list[dict[str, Any]]:
    filters = filters or {}
    categories = filter_categories(filters)
    if categories and len(categories) == 1:
        filters = filters_for_category(filters, categories[0])
    clauses: list[dict[str, Any]] = []

    if categories:
        clauses.append({"terms": {"category": categories}})

    for field in ("doc_id", "ticker"):
        values = _filter_values(filters.get(field))
        if values:
            clauses.append({"terms": {field: values}})

    years = _filter_values(filters.get("year", filters.get("fiscal_year")))
    if years:
        clauses.append({"terms": {"year": [int(value) for value in years]}})

    for field in ("source", "issuer"):
        value = filters.get(field)
        if value not in (None, "", [], ()):
            clauses.append({"match": {field: {"query": str(value)}}})

    return clauses


def _filter_values(value: Any) -> list[Any]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _filtered_categories(
    categories: list[str],
    filters: MetadataFilters | None,
) -> list[str]:
    filter_cats = filter_categories(filters)
    if not filter_cats:
        return categories
    allowed = set(filter_cats)
    return [category for category in categories if category in allowed]
