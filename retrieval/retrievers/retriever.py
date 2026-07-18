from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from retrieval.clients.embeddings import get_embed_model
from retrieval.clients.milvus_client import collection_name, create_milvus_client, milvus_enabled
from retrieval.clients.rerank_client import (
    rerank_documents,
    rerank_enabled,
    rerank_provider,
)
from retrieval.core.collections import (
    get_collection_registry,
    pdf_categories,
)
from retrieval.core.filters import (
    MetadataFilters,
    filter_categories,
    merge_filters,
)
from retrieval.core.kb_contract import RetrievalTrace, apply_on_empty_policy
from retrieval.retrievers.bm25 import bm25_scores

logger = get_logger(service="retriever")

DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.35
DEFAULT_RERANK_PARENT_EXCERPT_CHARS = 1600


@dataclass
class RetrievalHit:
    text: str
    score: float
    metadata: dict[str, Any]
    node_id: str | None = None
    category: str | None = None
    collection: str | None = None


class Retriever(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalHit]:
        raise NotImplementedError


class VectorRetriever(Retriever):
    """基于 Milvus 的 L3 向量检索；PG 只在召回后补 parent 上下文。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filters: MetadataFilters | None = None,
        candidate_multiplier: int = 4,
    ):
        registry = get_collection_registry()
        if categories is None:
            self.categories = list(registry.keys())
        else:
            unknown = [c for c in categories if c not in registry]
            if unknown:
                known = ", ".join(sorted(registry))
                raise ValueError(f"未知 categories={unknown}，可选: {known}")
            self.categories = list(categories)

        self.top_k = top_k
        self.similarity_threshold = (
            DEFAULT_VECTOR_SIMILARITY_THRESHOLD
            if similarity_threshold is None
            else similarity_threshold
        )
        self.metadata_filters = metadata_filters or {}
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self._client: Any | None = None
        self._embed_model: Any | None = None
        self.last_trace: RetrievalTrace | None = None

    def _ensure_milvus(self) -> bool:
        if not milvus_enabled():
            return False
        if self._client is None:
            self._client = create_milvus_client()
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return True

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: MetadataFilters | None = None,
        *,
        enforce_on_empty: bool = True,
    ) -> list[RetrievalHit]:
        k = top_k or self.top_k
        filters = merge_filters(self.metadata_filters, metadata_filters)
        categories = _filtered_categories(self.categories, filters)
        per_store_k = max(k, k * self.candidate_multiplier)
        try:
            milvus_ready = self._ensure_milvus()
        except Exception as exc:
            logger.warning("milvus vector search skipped error={}", exc)
            milvus_ready = False
        if not milvus_ready or self._client is None or self._embed_model is None:
            logger.warning("milvus vector search skipped reason=MILVUS_UNAVAILABLE")
            hits: list[RetrievalHit] = []
            if enforce_on_empty:
                hits, trace = apply_on_empty_policy(
                    hits,
                    query=query,
                    filters=_category_only_filters(filters),
                    categories=categories,
                    vector_hits=0,
                    lexical_hits=0,
                )
                self.last_trace = trace
            return hits

        query_embedding = self._embed_model.get_query_embedding(query)

        hits: list[RetrievalHit] = []
        for category in categories:
            name = collection_name(category)
            if not self._client.has_collection(name):
                logger.warning("milvus collection missing category={} collection={}", category, name)
                continue
            try:
                results = self._client.search(
                    collection_name=name,
                    data=[query_embedding],
                    limit=per_store_k,
                    output_fields=[
                        "chunk_id",
                        "doc_id",
                        "category",
                        "source",
                        "section_path",
                        "block_type",
                        "parent_chunk_id",
                        "root_chunk_id",
                        "chunk_index",
                        "page_num",
                        "text",
                        "metadata",
                    ],
                    search_params=_milvus_search_params(per_store_k),
                )
            except Exception as exc:
                logger.warning("milvus search failed category={} error={}", category, exc)
                continue

            for raw_hit in _flatten_milvus_results(results):
                score = _milvus_score(raw_hit)
                if self.similarity_threshold is not None and score < self.similarity_threshold:
                    continue
                entity = _milvus_entity(raw_hit)
                metadata = _coerce_metadata(entity.get("metadata"))
                metadata.setdefault("category", category)
                metadata.setdefault("collection", name)
                metadata.setdefault("chunk_id", entity.get("chunk_id") or raw_hit.get("id"))
                metadata.setdefault("doc_id", entity.get("doc_id"))
                metadata.setdefault("source", entity.get("source"))
                metadata.setdefault("section_path", entity.get("section_path"))
                metadata.setdefault("block_type", entity.get("block_type"))
                metadata.setdefault("parent_chunk_id", entity.get("parent_chunk_id"))
                metadata.setdefault("root_chunk_id", entity.get("root_chunk_id"))
                metadata.setdefault("chunk_index", entity.get("chunk_index"))
                metadata.setdefault("page_num", entity.get("page_num"))
                metadata["vector_score"] = score
                hits.append(
                    RetrievalHit(
                        text=str(entity.get("text") or ""),
                        score=score,
                        metadata=metadata,
                        node_id=str(metadata.get("chunk_id") or raw_hit.get("id") or "") or None,
                        category=category,
                        collection=name,
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = _hydrate_hits_from_parent_store(hits[:k])
        if enforce_on_empty:
            hits, trace = apply_on_empty_policy(
                hits,
                query=query,
                filters=_category_only_filters(filters),
                categories=categories,
                vector_hits=len(hits),
                lexical_hits=0,
            )
            self.last_trace = trace
        return hits


class HybridRetriever(Retriever):
    """Milvus 向量召回 + ES BM25 召回，本地 BM25 兜底，PG 补 parent 上下文。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filters: MetadataFilters | None = None,
        vector_weight: float = 0.65,
        candidate_top_k: int | None = None,
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
    ):
        self.categories = categories or list(get_collection_registry().keys())
        self.top_k = top_k
        self.metadata_filters = metadata_filters or {}
        self.vector_weight = min(max(vector_weight, 0.0), 1.0)
        self.candidate_top_k = candidate_top_k or max(top_k * 4, 20)
        fusion = str(fusion_mode or "rrf").strip().lower()
        if fusion not in {"rrf", "weighted"}:
            raise ValueError(f"unknown fusion_mode={fusion_mode!r}, expected 'rrf' or 'weighted'")
        self.fusion_mode = fusion
        self.rrf_k = max(int(rrf_k), 1)
        self.rerank_enabled = rerank_enabled()
        self.rerank_provider = rerank_provider() if self.rerank_enabled else None
        self.rerank_model = settings.RERANK_MODEL if self.rerank_enabled else None
        self.rerank_candidate_top_k = max(
            int(settings.RERANK_CANDIDATE_TOP_K or 0),
            self.top_k,
        )
        self.vector_retriever = VectorRetriever(
            categories=self.categories,
            top_k=self.candidate_top_k,
            similarity_threshold=similarity_threshold,
            metadata_filters=self.metadata_filters,
            candidate_multiplier=4,
        )
        self.es_bm25_retriever = None
        if settings.ELASTICSEARCH_ENABLED:
            from retrieval.retrievers.es_bm25 import ElasticsearchBM25Retriever

            self.es_bm25_retriever = ElasticsearchBM25Retriever(
                categories=self.categories,
                metadata_filters=self.metadata_filters,
            )
        self.last_trace: RetrievalTrace | None = None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalHit]:
        k = top_k or self.top_k
        filters = merge_filters(self.metadata_filters, metadata_filters)
        candidate_k = max(self.candidate_top_k, k * 4)
        active_categories = _filtered_categories(self.categories, filters)
        trace_filters = _category_only_filters(filters)

        vector_hits = self.vector_retriever.search(
            query,
            top_k=candidate_k,
            metadata_filters=filters,
            enforce_on_empty=False,
        )
        lexical_hits = self._lexical_search(
            query,
            top_k=candidate_k,
            metadata_filters=filters,
        )
        fusion_top_k = max(k, self.rerank_candidate_top_k) if self.rerank_enabled else k

        if not vector_hits and not lexical_hits:
            hits, trace = apply_on_empty_policy(
                [],
                query=query,
                filters=trace_filters,
                categories=active_categories,
                vector_hits=0,
                lexical_hits=0,
            )
            self.last_trace = trace
            if trace.abstained:
                logger.info(
                    "retrieval abstain reason={} policy={} filters={}",
                    trace.abstain_reason,
                    trace.on_empty_policy,
                    filters,
                )
            return hits

        if self.fusion_mode == "rrf":
            hits = _rrf_fuse_hits(
                [("vector", vector_hits), ("lexical", lexical_hits)],
                top_k=fusion_top_k,
                rrf_k=self.rrf_k,
            )
        else:
            hits = _weighted_fuse_hits(
                vector_hits,
                lexical_hits,
                top_k=fusion_top_k,
                vector_weight=self.vector_weight,
            )
        hits = _hydrate_hits_from_parent_store(hits)
        hits = self._rerank_hits(
            query,
            hits,
            top_k=k,
        )
        hits = self._select_diverse_hits(hits, top_k=k)
        hits, trace = apply_on_empty_policy(
            hits,
            query=query,
            filters=trace_filters,
            categories=active_categories,
            vector_hits=len(vector_hits),
            lexical_hits=len(lexical_hits),
        )
        if self.rerank_enabled:
            trace.extra.update(
                {
                    "rerank_provider": self.rerank_provider,
                    "rerank_model": self.rerank_model,
                }
            )
        trace.final_hits = len(hits)
        self.last_trace = trace
        return hits

    def _lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: MetadataFilters | None,
    ) -> list[RetrievalHit]:
        if self.es_bm25_retriever is not None:
            try:
                return self.es_bm25_retriever.search(
                    query,
                    top_k=top_k,
                    metadata_filters=metadata_filters,
                )
            except Exception as exc:
                logger.warning("es bm25 search failed error={}", exc)
        return self._local_bm25_search(
            query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

    def _local_bm25_search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: MetadataFilters | None,
    ) -> list[RetrievalHit]:
        filters = merge_filters(self.metadata_filters, metadata_filters)
        categories = _filtered_categories(self.categories, filters)
        candidates: list[RetrievalHit] = []
        for category in categories:
            try:
                candidates.extend(_load_pg_bm25_rows(category))
            except Exception as exc:
                logger.warning("local bm25 load failed category={} error={}", category, exc)
        return _bm25_rerank_hits(query, candidates, top_k=top_k)

    def _rerank_hits(
        self,
        query: str,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not self.rerank_enabled or not hits:
            return hits[:top_k]

        candidate_hits = hits[: self.rerank_candidate_top_k]
        try:
            reranked = rerank_documents(
                query=query,
                documents=[_rerank_text(hit) for hit in candidate_hits],
                top_n=len(candidate_hits),
            )
        except Exception as exc:
            logger.warning(
                "external rerank failed provider={} model={} error={}",
                self.rerank_provider,
                self.rerank_model,
                exc,
            )
            return hits[:top_k]

        if not reranked:
            return hits[:top_k]

        ordered: list[RetrievalHit] = []
        seen: set[int] = set()
        for item in reranked:
            if item.index < 0 or item.index >= len(candidate_hits):
                continue
            seen.add(item.index)
            hit = candidate_hits[item.index]
            metadata = dict(hit.metadata)
            metadata["rerank_score"] = item.score
            metadata["rerank_provider"] = self.rerank_provider
            metadata["rerank_model"] = self.rerank_model
            ordered.append(
                RetrievalHit(
                    text=hit.text,
                    score=item.score,
                    metadata=metadata,
                    node_id=hit.node_id,
                    category=hit.category,
                    collection=hit.collection,
                )
            )

        if not ordered:
            return hits[:top_k]

        for index, hit in enumerate(candidate_hits):
            if index in seen:
                continue
            ordered.append(hit)

        return ordered[:top_k]

    def _select_diverse_hits(
        self,
        hits: list[RetrievalHit],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if not hits:
            return []

        selected: list[RetrievalHit] = []
        selected_keys: set[str] = set()
        remaining = list(hits)
        passes = (
            {
                "max_chunks_per_doc": 2,
                "enforce_page_limit": True,
                "enforce_adjacent_limit": True,
            },
            {
                "max_chunks_per_doc": 2,
                "enforce_page_limit": False,
                "enforce_adjacent_limit": False,
            },
            {
                "max_chunks_per_doc": None,
                "enforce_page_limit": False,
                "enforce_adjacent_limit": False,
            },
        )

        for rules in passes:
            if len(selected) >= top_k or not remaining:
                break
            next_remaining: list[RetrievalHit] = []
            for hit in remaining:
                if len(selected) >= top_k:
                    next_remaining.append(hit)
                    continue
                hit_key = _hit_key(hit)
                if hit_key in selected_keys:
                    continue
                if self._violates_diversity_rules(
                    hit,
                    selected,
                    max_chunks_per_doc=rules["max_chunks_per_doc"],
                    enforce_page_limit=rules["enforce_page_limit"],
                    enforce_adjacent_limit=rules["enforce_adjacent_limit"],
                ):
                    next_remaining.append(hit)
                    continue
                selected.append(hit)
                selected_keys.add(hit_key)
            remaining = next_remaining

        if len(selected) >= top_k:
            return selected[:top_k]

        for hit in remaining:
            if len(selected) >= top_k:
                break
            hit_key = _hit_key(hit)
            if hit_key in selected_keys:
                continue
            selected.append(hit)
            selected_keys.add(hit_key)
        return selected[:top_k]

    def _violates_diversity_rules(
        self,
        candidate: RetrievalHit,
        selected: list[RetrievalHit],
        *,
        max_chunks_per_doc: int | None,
        enforce_page_limit: bool,
        enforce_adjacent_limit: bool,
    ) -> bool:
        candidate_doc_id = _metadata_string(candidate.metadata, "doc_id")
        candidate_page_num = _metadata_int(candidate.metadata, "page_num")
        candidate_chunk_index = _metadata_int(candidate.metadata, "chunk_index")

        if candidate_doc_id and max_chunks_per_doc is not None:
            doc_hits = sum(
                1
                for hit in selected
                if _metadata_string(hit.metadata, "doc_id") == candidate_doc_id
            )
            if doc_hits >= max_chunks_per_doc:
                return True

        if not candidate_doc_id:
            return False

        for hit in selected:
            if _metadata_string(hit.metadata, "doc_id") != candidate_doc_id:
                continue

            if enforce_page_limit:
                selected_page_num = _metadata_int(hit.metadata, "page_num")
                if (
                    candidate_page_num is not None
                    and selected_page_num is not None
                    and candidate_page_num == selected_page_num
                ):
                    return True

            if enforce_adjacent_limit:
                selected_chunk_index = _metadata_int(hit.metadata, "chunk_index")
                if (
                    candidate_chunk_index is not None
                    and selected_chunk_index is not None
                    and abs(candidate_chunk_index - selected_chunk_index) <= 1
                ):
                    return True

        return False

# 兼容旧名
FAQRetriever = Retriever
VectorFAQRetriever = VectorRetriever


def get_retriever(
    categories: list[str] | None = None,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    metadata_filters: MetadataFilters | None = None,
    hybrid: bool = False,
    fusion_mode: str = "rrf",
    rrf_k: int = 60,
    vector_weight: float = 0.65,
) -> Retriever:
    if hybrid:
        return HybridRetriever(
            categories=categories,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filters=metadata_filters,
            fusion_mode=fusion_mode,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
        )
    return VectorRetriever(
        categories=categories,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metadata_filters=metadata_filters,
    )


def get_faq_retriever(
    top_k: int = 5,
    similarity_threshold: float | None = None,
) -> Retriever:
    """仅检索 FAQ Markdown 集合。"""
    return get_retriever(
        categories=["faq"],
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def get_pdf_retriever(
    categories: list[str] | None = None,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    metadata_filters: MetadataFilters | None = None,
    hybrid: bool = True,
) -> Retriever:
    """检索 PDF 切块集合（默认全部 PDF 类别，不含 FAQ）。"""
    return get_retriever(
        categories=categories or pdf_categories(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metadata_filters=metadata_filters,
        hybrid=hybrid,
    )


def _filtered_categories(
    categories: list[str],
    filters: MetadataFilters | None,
) -> list[str]:
    filter_cats = filter_categories(filters)
    if not filter_cats:
        return categories
    allowed = set(filter_cats)
    return [category for category in categories if category in allowed]


def _category_only_filters(filters: MetadataFilters | None) -> MetadataFilters:
    categories = filter_categories(filters)
    return {"category": categories} if categories else {}


def _hit_key(hit: RetrievalHit) -> str:
    if hit.node_id:
        return f"node:{hit.node_id}"
    meta = hit.metadata
    return "|".join(
        str(meta.get(field, ""))
        for field in ("category", "source", "page_num", "chunk_index")
    ) or hit.text[:80]


def _bm25_rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    if not hits:
        return []
    scores = bm25_scores(query, [hit.text for hit in hits])
    ranked: list[RetrievalHit] = []
    for hit, score in zip(hits, scores, strict=False):
        if score <= 0:
            continue
        metadata = dict(hit.metadata)
        metadata["bm25_score"] = score
        metadata["bm25_backend"] = "local"
        ranked.append(
            RetrievalHit(
                text=hit.text,
                score=score,
                metadata=metadata,
                node_id=hit.node_id,
                category=hit.category,
                collection=hit.collection,
            )
        )
    ranked.sort(key=lambda hit: hit.score, reverse=True)
    return ranked[:top_k]


def _rrf_fuse_hits(
    ranked_lists: list[tuple[str, list[RetrievalHit]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion over one or more ranked hit lists."""
    scores: dict[str, float] = {}
    by_key: dict[str, RetrievalHit] = {}
    ranks_by_key: dict[str, dict[str, int]] = {}

    for list_name, ranked in ranked_lists:
        for rank, hit in enumerate(ranked, 1):
            key = _hit_key(hit)
            by_key.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            ranks_by_key.setdefault(key, {})[list_name] = rank

    fused: list[RetrievalHit] = []
    for key, score in scores.items():
        hit = by_key[key]
        metadata = dict(hit.metadata)
        metadata["rrf_score"] = score
        metadata["hybrid_score"] = score
        list_ranks = ranks_by_key.get(key) or {}
        if "vector" in list_ranks:
            metadata["vector_rank"] = list_ranks["vector"]
        if "lexical" in list_ranks:
            metadata["bm25_rank"] = list_ranks["lexical"]
        fused.append(
            RetrievalHit(
                text=hit.text,
                score=score,
                metadata=metadata,
                node_id=hit.node_id,
                category=hit.category,
                collection=hit.collection,
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


def _weighted_fuse_hits(
    vector_hits: list[RetrievalHit],
    lexical_hits: list[RetrievalHit],
    *,
    top_k: int,
    vector_weight: float,
) -> list[RetrievalHit]:
    lexical_weight = 1.0 - vector_weight
    vector_norm = _normalized_by_key(vector_hits)
    lexical_norm = _normalized_by_key(lexical_hits)
    by_key: dict[str, RetrievalHit] = {}

    for hit in [*vector_hits, *lexical_hits]:
        by_key.setdefault(_hit_key(hit), hit)

    fused: list[RetrievalHit] = []
    for key, hit in by_key.items():
        v_score = vector_norm.get(key)
        b_score = lexical_norm.get(key)
        if v_score is None:
            score = max(0.5 * (b_score or 0.0), lexical_weight * (b_score or 0.0))
        elif b_score is None:
            score = max(0.7 * v_score, vector_weight * v_score)
        else:
            score = vector_weight * v_score + lexical_weight * b_score

        metadata = dict(hit.metadata)
        if v_score is not None:
            metadata["vector_score_norm"] = v_score
        if b_score is not None:
            metadata["bm25_score_norm"] = b_score
        metadata["hybrid_score"] = score
        fused.append(
            RetrievalHit(
                text=hit.text,
                score=score,
                metadata=metadata,
                node_id=hit.node_id,
                category=hit.category,
                collection=hit.collection,
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


def _normalized_by_key(hits: list[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {}
    max_score = max((h.score for h in hits), default=0.0)
    if max_score <= 0:
        return {}
    return {_hit_key(hit): hit.score / max_score for hit in hits}


def _flatten_milvus_results(results: Any) -> list[dict[str, Any]]:
    if not results:
        return []
    if isinstance(results, list) and results and isinstance(results[0], list):
        return [dict(item) for item in results[0]]
    if isinstance(results, list):
        return [dict(item) for item in results]
    return []


def _milvus_entity(raw_hit: dict[str, Any]) -> dict[str, Any]:
    entity = raw_hit.get("entity") or raw_hit.get("fields") or {}
    return dict(entity) if isinstance(entity, dict) else {}


def _milvus_score(raw_hit: dict[str, Any]) -> float:
    value = raw_hit.get("distance", raw_hit.get("score", raw_hit.get("similarity", 0.0)))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _milvus_search_params(limit: int) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if settings.MILVUS_INDEX_TYPE.upper() == "HNSW":
        params["ef"] = max(int(settings.MILVUS_SEARCH_EF or 0), int(limit))
    return {
        "metric_type": settings.MILVUS_METRIC_TYPE,
        "params": params,
    }


def _rerank_text(hit: RetrievalHit) -> str:
    text = str(hit.metadata.get("rerank_text") or "").strip()
    return text or hit.text


def _build_rerank_text(leaf_text: str, parent_text: str) -> str:
    leaf = " ".join(str(leaf_text or "").split())
    parent = " ".join(str(parent_text or "").split())
    if not parent or parent == leaf:
        return leaf
    excerpt = parent[:DEFAULT_RERANK_PARENT_EXCERPT_CHARS]
    return f"{leaf}\n\n上下文摘录：\n{excerpt}"


def _metadata_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if value is not None else ""


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hydrate_hits_from_parent_store(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    if not hits:
        return hits

    chunk_ids = [
        str(hit.node_id or hit.metadata.get("chunk_id") or "").strip()
        for hit in hits
    ]
    chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    if not chunk_ids:
        return hits

    contexts = _load_parent_contexts(chunk_ids)
    if not contexts:
        return hits

    hydrated: list[RetrievalHit] = []
    for hit in hits:
        chunk_id = str(hit.node_id or hit.metadata.get("chunk_id") or "").strip()
        context = contexts.get(chunk_id)
        if context is None:
            hydrated.append(hit)
            continue

        metadata = dict(hit.metadata)
        metadata.update(context["leaf_metadata"])
        leaf_text = hit.text
        parent_text = str(context.get("context_text") or hit.text)
        metadata["leaf_text"] = leaf_text
        metadata["rerank_text"] = _build_rerank_text(leaf_text, parent_text)
        metadata["context_chunk_id"] = context.get("context_chunk_id")
        metadata["context_chunk_level"] = context.get("context_chunk_level")
        metadata["context_section_path"] = context.get("context_section_path")
        metadata["context_page_range"] = context.get("context_page_range")
        hydrated.append(
            RetrievalHit(
                text=parent_text,
                score=hit.score,
                metadata=metadata,
                node_id=hit.node_id,
                category=hit.category,
                collection=hit.collection,
            )
        )
    return hydrated


def _load_parent_contexts(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    import psycopg
    from psycopg import sql

    from retrieval.indexing.index import _pg_connection_strings
    from retrieval.indexing.parent_store import RAG_SCHEMA

    sync_url, _ = _pg_connection_strings()

    contexts: dict[str, dict[str, Any]] = {}
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        cr.chunk_id,
                        cr.metadata,
                        pc.chunk_id,
                        pc.chunk_level,
                        pc.text,
                        pc.section_path,
                        pc.page_range,
                        pc.metadata
                    FROM {}.chunk_registry cr
                    LEFT JOIN {}.parent_chunks pc
                        ON pc.chunk_id = COALESCE(NULLIF(cr.parent_chunk_id, ''), NULLIF(cr.root_chunk_id, ''))
                    WHERE cr.chunk_id = ANY(%s)
                    """
                ).format(
                    sql.Identifier(RAG_SCHEMA),
                    sql.Identifier(RAG_SCHEMA),
                ),
                (chunk_ids,),
            )
            for (
                leaf_chunk_id,
                leaf_metadata,
                context_chunk_id,
                context_chunk_level,
                context_text,
                context_section_path,
                context_page_range,
                context_metadata,
            ) in cur.fetchall():
                contexts[str(leaf_chunk_id)] = {
                    "leaf_metadata": _coerce_metadata(leaf_metadata),
                    "context_metadata": _coerce_metadata(context_metadata),
                    "context_chunk_id": context_chunk_id,
                    "context_chunk_level": context_chunk_level,
                    "context_text": context_text,
                    "context_section_path": context_section_path,
                    "context_page_range": context_page_range,
                }
    return contexts


def _load_pg_bm25_rows(category: str) -> list[RetrievalHit]:
    import psycopg
    from psycopg import sql

    from retrieval.indexing.index import _pg_connection_strings
    from retrieval.indexing.parent_store import RAG_SCHEMA

    sync_url, _ = _pg_connection_strings()
    rows: list[RetrievalHit] = []
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT chunk_id, text, metadata
                    FROM {}.parent_chunks
                    WHERE category = %s
                      AND chunk_level = 'L2'
                    """
                ).format(sql.Identifier(RAG_SCHEMA)),
                (category,),
            )
            for chunk_id, text, metadata in cur.fetchall():
                meta = _coerce_metadata(metadata)
                row_text = str(text or "")
                if not row_text:
                    continue
                meta.setdefault("category", category)
                meta.setdefault("chunk_id", chunk_id)
                meta.setdefault("chunk_level", "L2")
                rows.append(
                    RetrievalHit(
                        text=row_text,
                        score=0.0,
                        metadata=meta,
                        node_id=str(chunk_id),
                        category=category,
                        collection=str(meta.get("collection") or ""),
                    )
                )
    return rows


def _coerce_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
