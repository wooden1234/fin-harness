from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from retrieval.clients.rerank_client import rerank_documents, rerank_enabled
from retrieval.retrievers.bm25 import bm25_scores
from retrieval.core.collections import (
    get_collection_registry,
    get_table_name,
    pdf_categories,
)
from retrieval.core.filters import (
    MetadataFilters,
    filter_categories,
    has_strict_filters,
    merge_filters,
    metadata_matches,
)
from retrieval.core.kb_contract import RetrievalTrace, apply_on_empty_policy
from retrieval.indexing.index import load_index

logger = get_logger(service="retriever")


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
    """单库或多库向量检索；多库时按 score 合并取 Top-K。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float | None = 0.5,
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
        self.similarity_threshold = similarity_threshold
        self.metadata_filters = metadata_filters or {}
        self.candidate_multiplier = max(candidate_multiplier, 1)
        self._indices = {cat: load_index(cat) for cat in self.categories}
        self.last_trace: RetrievalTrace | None = None

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
        if has_strict_filters(filters):
            # Exact filters need a deeper vector scan before post-filtering.
            per_store_k = max(per_store_k, k * 12)

        hits: list[RetrievalHit] = []
        for category in categories:
            index = self._indices[category]
            retriever = index.as_retriever(similarity_top_k=per_store_k)
            for nws in retriever.retrieve(query):
                score = float(nws.score or 0.0)
                if self.similarity_threshold is not None and score < self.similarity_threshold:
                    continue
                node = nws.node
                metadata = dict(node.metadata or {})
                metadata.setdefault("category", category)
                metadata.setdefault("collection", get_table_name(category))
                if not metadata_matches(metadata, filters):
                    continue
                hits.append(
                    RetrievalHit(
                        text=node.get_content(metadata_mode="none"),
                        score=score,
                        metadata=metadata,
                        node_id=node.node_id,
                        category=category,
                        collection=get_table_name(category),
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]
        if enforce_on_empty:
            hits, trace = apply_on_empty_policy(
                hits,
                query=query,
                filters=filters,
                categories=categories,
                vector_hits=len(hits),
                lexical_hits=0,
            )
            self.last_trace = trace
        return hits


class HybridRetriever(Retriever):
    """Vector retrieval plus BM25 lexical retrieval over the same collections."""

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
                filters=filters,
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
        hits = self._rerank_hits(
            query,
            hits,
            top_k=k,
        )
        hits, trace = apply_on_empty_policy(
            hits,
            query=query,
            filters=filters,
            categories=active_categories,
            vector_hits=len(vector_hits),
            lexical_hits=len(lexical_hits),
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
        return self._bm25_search(
            query,
            top_k=top_k,
            metadata_filters=metadata_filters,
        )

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
                documents=[hit.text for hit in candidate_hits],
                top_n=min(top_k, len(candidate_hits)),
            )
        except Exception as exc:
            logger.warning("external rerank failed error={}", exc)
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
            ordered.append(
                RetrievalHit(
                    text=item.document or hit.text,
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

    def _bm25_search(
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
                rows = _load_pg_text_rows(category)
            except Exception as exc:
                logger.warning("bm25 load failed category={} error={}", category, exc)
                continue
            for hit in rows:
                if metadata_matches(hit.metadata, filters):
                    candidates.append(hit)

        if not candidates:
            return []
        return _bm25_rerank_hits(query, candidates, top_k=top_k)


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
    scores = bm25_scores(query, [h.text for h in hits])
    ranked: list[RetrievalHit] = []
    for hit, score in zip(hits, scores, strict=False):
        if score <= 0:
            continue
        metadata = dict(hit.metadata)
        metadata["bm25_score"] = score
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
    ranked.sort(key=lambda h: h.score, reverse=True)
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


@lru_cache(maxsize=16)
def _load_pg_text_rows(category: str) -> tuple[RetrievalHit, ...]:
    import psycopg
    from psycopg import sql

    from retrieval.indexing.index import VECTOR_SCHEMA, _pg_connection_strings

    table_name = get_table_name(category)
    physical_table = f"data_{table_name}"
    sync_url, _ = _pg_connection_strings()

    rows: list[RetrievalHit] = []
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT node_id, text, metadata_ FROM {}.{}").format(
                    sql.Identifier(VECTOR_SCHEMA),
                    sql.Identifier(physical_table)
                )
            )
            for node_id, text, metadata in cur.fetchall():
                meta = _coerce_metadata(metadata)
                meta.setdefault("category", category)
                meta.setdefault("collection", table_name)
                rows.append(
                    RetrievalHit(
                        text=text or "",
                        score=0.0,
                        metadata=meta,
                        node_id=str(node_id) if node_id is not None else None,
                        category=category,
                        collection=table_name,
                    )
                )
    return tuple(rows)


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
