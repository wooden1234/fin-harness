"""Elasticsearch BM25 词法检索。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from retrieval.collections import get_collection_registry, get_table_name
from retrieval.es_client import create_es_client, index_name
from retrieval.filters import MetadataFilters, filter_categories, merge_filters, metadata_matches

if TYPE_CHECKING:
    from retrieval.retriever import RetrievalHit

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
        from retrieval.retriever import RetrievalHit

        filters = merge_filters(self.metadata_filters, metadata_filters)
        categories = _filtered_categories(self.categories, filters)
        if not categories:
            return []

        response = self._client.search(
            index=[index_name(category) for category in categories],
            body={
                "query": {
                    "bool": {
                        "must": [_text_query(query)],
                        "filter": _filter_clauses(filters),
                    }
                },
                "size": top_k,
            },
        )

        hits: list[RetrievalHit] = []
        for raw_hit in response.get("hits", {}).get("hits", []):
            source = raw_hit.get("_source") or {}
            metadata = _hit_metadata(source)
            if not metadata_matches(metadata, filters):
                continue
            category = str(source.get("category") or metadata.get("category") or "")
            collection = str(
                source.get("collection")
                or metadata.get("collection")
                or (get_table_name(category) if category else "")
            )
            metadata.setdefault("category", category)
            metadata.setdefault("collection", collection)
            metadata["bm25_score"] = float(raw_hit.get("_score") or 0.0)
            metadata["es_index"] = raw_hit.get("_index")
            hits.append(
                RetrievalHit(
                    text=str(source.get("text") or ""),
                    score=float(raw_hit.get("_score") or 0.0),
                    metadata=metadata,
                    node_id=str(source.get("node_id") or "") or None,
                    category=category or None,
                    collection=collection or None,
                )
            )
        return hits


def _hit_metadata(source: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(source.get("metadata") or {})
    # 根字段是建索引时提升出来的权威值（含 doc_id）
    for key in _ROOT_META_FIELDS:
        value = source.get(key)
        if value is not None and value != "":
            metadata[key] = value
    return metadata


def _text_query(query: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": query,
            "fields": ["text^3", "title^2", "section^1.5", "source"],
            "type": "best_fields",
            "operator": "or",
        }
    }


def _filter_clauses(filters: MetadataFilters | None) -> list[dict[str, Any]]:
    filters = filters or {}
    clauses: list[dict[str, Any]] = []

    categories = filter_categories(filters)
    if categories:
        clauses.append({"terms": {"category": categories}})

    doc_id = filters.get("doc_id")
    if doc_id:
        clauses.append({"term": {"doc_id": str(doc_id)}})

    ticker = filters.get("ticker")
    if ticker:
        clauses.append({"term": {"ticker": str(ticker)}})

    year = filters.get("year") or filters.get("fiscal_year")
    if year is not None:
        year_int = _to_int(year)
        clauses.append(
            {
                "bool": {
                    "should": [
                        {"term": {"year": year_int}},
                        {"term": {"fiscal_year": year_int}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    source = filters.get("source")
    if source:
        clauses.append(
            {
                "multi_match": {
                    "query": str(source),
                    "fields": ["source", "file", "title", "doc_id"],
                }
            }
        )

    issuer = filters.get("issuer")
    if issuer:
        clauses.append(
            {
                "multi_match": {
                    "query": str(issuer),
                    "fields": ["issuer", "title"],
                }
            }
        )

    company = filters.get("company")
    if company:
        clauses.append(
            {
                "multi_match": {
                    "query": str(company),
                    "fields": [
                        "company",
                        "ticker",
                        "source",
                        "file",
                        "title",
                        "issuer",
                        "doc_group",
                    ],
                }
            }
        )

    return clauses


def _filtered_categories(
    categories: list[str],
    filters: MetadataFilters | None,
) -> list[str]:
    filter_cats = filter_categories(filters)
    if not filter_cats:
        return categories
    allowed = set(filter_cats)
    return [category for category in categories if category in allowed]


def _to_int(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)
