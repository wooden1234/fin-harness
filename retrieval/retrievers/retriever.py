from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
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
    filters_for_category,
    merge_filters,
    metadata_matches,
)
from retrieval.core.kb_contract import RetrievalTrace, apply_on_empty_policy
from retrieval.retrievers.bm25 import bm25_scores

logger = get_logger(service="retriever")

DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.35
AUTO_MERGE_SCORE_BONUS = 0.02


@dataclass
class RetrievalHit:
    text: str
    score: float
    metadata: dict[str, Any]
    node_id: str | None = None
    category: str | None = None
    collection: str | None = None
    score_type: str = "unknown"


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
    """基于 Milvus 的 L3 向量检索。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filters: MetadataFilters | None = None,
        candidate_multiplier: int | None = None,
        diversify: bool = True,
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
        configured_multiplier = (
            settings.VECTOR_CANDIDATE_MULTIPLIER
            if candidate_multiplier is None
            else candidate_multiplier
        )
        self.candidate_multiplier = max(int(configured_multiplier), 1)
        self.diversify = diversify
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
            category_filters = filters_for_category(filters, category)
            search_kwargs: dict[str, Any] = {
                "collection_name": name,
                "data": [query_embedding],
                "limit": per_store_k,
                "output_fields": [
                    "chunk_id",
                    "doc_id",
                    "ticker",
                    "issuer",
                    "fiscal_year",
                    "year",
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
                "search_params": _milvus_search_params(per_store_k),
            }
            filter_expr = _milvus_filter_expr(category_filters)
            if filter_expr:
                search_kwargs["filter"] = filter_expr
            try:
                results = self._client.search(**search_kwargs)
            except Exception as exc:
                logger.warning(
                    "milvus filtered search failed category={} error={}",
                    category,
                    type(exc).__name__,
                )
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
                metadata.setdefault("ticker", entity.get("ticker"))
                metadata.setdefault("issuer", entity.get("issuer"))
                metadata.setdefault("fiscal_year", entity.get("fiscal_year"))
                metadata.setdefault("year", entity.get("year"))
                metadata.setdefault("source", entity.get("source"))
                metadata.setdefault("section_path", entity.get("section_path"))
                metadata.setdefault("block_type", entity.get("block_type"))
                metadata.setdefault("parent_chunk_id", entity.get("parent_chunk_id"))
                metadata.setdefault("root_chunk_id", entity.get("root_chunk_id"))
                metadata.setdefault("chunk_index", entity.get("chunk_index"))
                metadata.setdefault("page_num", entity.get("page_num"))
                if not metadata_matches(metadata, category_filters):
                    continue
                metadata["vector_score"] = score
                hits.append(
                    RetrievalHit(
                        text=str(entity.get("text") or ""),
                        score=score,
                        metadata=metadata,
                        node_id=str(metadata.get("chunk_id") or raw_hit.get("id") or "") or None,
                        category=category,
                        collection=name,
                        score_type="vector",
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        if self.diversify:
            hits = _select_vector_hits(hits, top_k=k)
        else:
            hits = hits[:k]
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
    """Milvus 向量召回 + ES BM25 召回，本地 BM25 兜底。"""

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
        rerank_min_score: float | None = None,
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
        configured_min_score = (
            settings.RERANK_MIN_SCORE
            if rerank_min_score is None
            else rerank_min_score
        )
        self.rerank_min_score = max(float(configured_min_score or 0.0), 0.0)
        self.last_rerank_status = "not_run"
        self.last_rerank_error = ""
        self.last_rerank_hits: list[RetrievalHit] = []
        self.vector_retriever = VectorRetriever(
            categories=self.categories,
            top_k=self.candidate_top_k,
            similarity_threshold=similarity_threshold,
            metadata_filters=self.metadata_filters,
            candidate_multiplier=4,
            # hybrid 先保留完整向量候选，避免在 rerank 前淘汰低排名但有效的 chunk。
            diversify=False,
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
        self.last_rerank_status = "not_run"
        self.last_rerank_error = ""
        self.last_rerank_hits = []
        filters = merge_filters(self.metadata_filters, metadata_filters)
        candidate_k = max(self.candidate_top_k, k * 4)
        active_categories = _filtered_categories(self.categories, filters)
        trace_filters = _category_only_filters(filters)

        vector_hits = self.vector_retriever.search(
            query,
            top_k=candidate_k,
            metadata_filters=filters,
            enforce_on_empty=False,
        ) or []
        lexical_hits = self._lexical_search(
            query,
            top_k=candidate_k,
            metadata_filters=filters,
        ) or []
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
            trace.extra.update(
                {
                    "rerank_status": "no_candidates",
                    "rerank_error": "",
                    "score_source": self.fusion_mode,
                }
            )
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
        # rerank 先确定最终 top-k，之后只对这些命中做父节点合并。
        hits = self._rerank_hits(query, hits, top_k=k)
        # 评估需要使用 rerank 后的叶子 chunk，避免父节点合并改变 chunk/page 召回口径。
        self.last_rerank_hits = list(hits)
        hits = _auto_merge_parent_hits(hits, top_k=k)
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
        trace.extra.update(
            {
                "rerank_status": self.last_rerank_status,
                "rerank_error": self.last_rerank_error,
                "score_source": "rerank"
                if self.last_rerank_status == "success"
                else self.fusion_mode,
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
        def fallback(reason: str, status: str = "fallback_to_fusion") -> list[RetrievalHit]:
            self.last_rerank_status = status
            self.last_rerank_error = reason
            selected = hits[:top_k]
            for hit in selected:
                hit.metadata["rerank_status"] = status
                hit.metadata["rerank_fallback_reason"] = reason
                hit.metadata["score_source"] = hit.score_type or self.fusion_mode
            return selected

        self.last_rerank_error = ""
        if not self.rerank_enabled or not hits:
            return fallback("rerank_disabled" if hits else "no_candidates", "disabled" if hits else "no_candidates")

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
            return fallback(type(exc).__name__)

        if not reranked:
            return fallback("empty_rerank_result")

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
            metadata["rerank_status"] = "success"
            metadata["score_source"] = "rerank"
            ordered.append(
                RetrievalHit(
                    text=hit.text,
                    score=item.score,
                    metadata=metadata,
                    node_id=hit.node_id,
                    category=hit.category,
                    collection=hit.collection,
                    score_type="rerank",
                )
            )

        if not ordered:
            return fallback("invalid_rerank_result")

        # 只在整个查询的最高相关度不足时拒绝返回，避免逐条截断破坏可回答问题的上下文召回。
        if self.rerank_min_score > 0.0 and ordered[0].score < self.rerank_min_score:
            self.last_rerank_status = "below_threshold"
            self.last_rerank_error = (
                f"top_score={ordered[0].score:.6f} min_score={self.rerank_min_score:.6f}"
            )
            return []

        for index, hit in enumerate(candidate_hits):
            if index in seen:
                continue
            ordered.append(hit)

        self.last_rerank_status = "success"
        return ordered[:top_k]

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
    rerank_min_score: float | None = None,
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
            rerank_min_score=rerank_min_score,
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
    rerank_min_score: float | None = None,
) -> Retriever:
    """检索 PDF 切块集合（默认全部 PDF 类别，不含 FAQ）。"""
    return get_retriever(
        categories=categories or pdf_categories(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metadata_filters=metadata_filters,
        hybrid=hybrid,
        rerank_min_score=rerank_min_score,
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


def _scale_vector_diversity_penalty(
    penalty: float,
    *,
    quality_hit_count: int,
    top_k: int,
) -> float:
    """高质量候选不足时，按可用质量候选比例降低多样性惩罚。"""
    if top_k <= 0:
        return 0.0
    ratio = min(max(quality_hit_count / top_k, 0.0), 1.0)
    return penalty * ratio


def _select_vector_hits(hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]:
    """在向量候选中保留相关性，同时根据当前候选分布软化文档重复。"""
    if not hits or top_k <= 0:
        return []

    unique_hits: list[RetrievalHit] = []
    seen_keys: set[str] = set()
    for hit in hits:
        key = _hit_key(hit)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_hits.append(hit)

    if len(unique_hits) <= top_k or not settings.VECTOR_DIVERSITY_ENABLED:
        return unique_hits[:top_k]

    doc_counts: dict[str, int] = {}
    for hit in unique_hits:
        doc_key = _vector_doc_key(hit)
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1

    candidate_count = len(unique_hits)
    unique_doc_count = len(doc_counts)
    if unique_doc_count < 2:
        return unique_hits[:top_k]

    duplicate_rate = 1.0 - unique_doc_count / candidate_count
    target_rate = min(max(settings.VECTOR_DIVERSITY_TARGET_DUPLICATE_RATE, 0.0), 0.99)
    pressure = max(0.0, duplicate_rate - target_rate) / max(1.0 - target_rate, 1e-6)
    concentration = sum((count / candidate_count) ** 2 for count in doc_counts.values())
    penalty = settings.VECTOR_DIVERSITY_STRENGTH * pressure * (1.0 + concentration)
    penalty = min(max(penalty, 0.0), max(settings.VECTOR_DIVERSITY_MAX_PENALTY, 0.0))
    if penalty <= 0.0:
        return unique_hits[:top_k]

    top_score = unique_hits[0].score
    bottom_score = unique_hits[-1].score
    score_span = top_score - bottom_score
    quality_ratio = min(
        max(settings.VECTOR_DIVERSITY_MIN_SCORE_RATIO, 0.0),
        1.0,
    )
    quality_hit_count = top_k
    if top_score > 0.0:
        quality_hits = [
            hit
            for hit in unique_hits
            if hit.score >= top_score * quality_ratio
        ]
        quality_hit_count = len(quality_hits)
        if len(quality_hits) >= top_k:
            unique_hits = quality_hits
            bottom_score = unique_hits[-1].score
            score_span = top_score - bottom_score

    penalty = _scale_vector_diversity_penalty(
        penalty,
        quality_hit_count=quality_hit_count,
        top_k=top_k,
    )

    scored_hits = [
        (
            hit,
            1.0 if score_span <= 0 else (hit.score - bottom_score) / score_span,
        )
        for hit in unique_hits
    ]

    selected: list[RetrievalHit] = []
    selected_counts: dict[str, int] = {}
    remaining = list(scored_hits)
    while remaining and len(selected) < top_k:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                remaining[index][1]
                - penalty * math.log1p(selected_counts.get(_vector_doc_key(remaining[index][0]), 0)),
                remaining[index][0].score,
            ),
        )
        hit, _ = remaining.pop(best_index)
        selected.append(hit)
        doc_key = _vector_doc_key(hit)
        selected_counts[doc_key] = selected_counts.get(doc_key, 0) + 1

    return selected


def _vector_doc_key(hit: RetrievalHit) -> str:
    doc_id = str(hit.metadata.get("doc_id") or "").strip()
    return f"doc:{doc_id}" if doc_id else f"chunk:{_hit_key(hit)}"


def _parent_merge_key(hit: RetrievalHit) -> str:
    """优先使用 L2 父块；没有父块时退回根块或子块自身。"""
    metadata = hit.metadata
    parent_id = str(metadata.get("parent_chunk_id") or "").strip()
    if parent_id:
        return f"parent:{parent_id}"
    root_id = str(metadata.get("root_chunk_id") or "").strip()
    if root_id:
        return f"root:{root_id}"
    return f"child:{_hit_key(hit)}"


def _auto_merge_parent_hits(
    hits: list[RetrievalHit],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    """读取 rerank top-k 对应父节点，并合并同父节点的子块证据。"""
    if not hits or top_k <= 0:
        return []

    groups: dict[str, list[RetrievalHit]] = {}
    order: list[str] = []
    for hit in hits:
        key = _parent_merge_key(hit)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(hit)

    min_children = max(int(settings.AUTO_MERGE_MIN_CHILDREN or 2), 2)
    mergeable_keys = {
        key
        for key, children in groups.items()
        if len(children) >= min_children and key.startswith(("parent:", "root:"))
    }
    parent_ids = [
        key.split(":", 1)[1]
        for key in mergeable_keys
    ]
    parent_nodes = _load_parent_nodes(parent_ids)

    merged: list[RetrievalHit] = []
    for key in order:
        children = groups[key]
        if key not in mergeable_keys:
            for child in children:
                metadata = dict(child.metadata)
                child_id = str(child.node_id or metadata.get("chunk_id") or "").strip()
                metadata["auto_merged"] = False
                metadata["child_chunk_ids"] = [child_id] if child_id else []
                metadata["evidence_child_ids"] = [child_id] if child_id else []
                metadata["auto_merge_child_count"] = 1
                merged.append(
                    RetrievalHit(
                        text=child.text,
                        score=child.score,
                        metadata=metadata,
                        node_id=child.node_id,
                    category=child.category,
                    collection=child.collection,
                    score_type=child.score_type,
                )
                )
            continue
        representative = children[0]
        child_ids = [
            str(child.node_id or child.metadata.get("chunk_id") or "").strip()
            for child in children
        ]
        child_ids = list(dict.fromkeys(child_id for child_id in child_ids if child_id))
        metadata = dict(representative.metadata)
        metadata["auto_merged"] = len(children) > 1
        metadata["child_chunk_ids"] = child_ids
        metadata["evidence_child_ids"] = child_ids
        metadata["auto_merge_child_count"] = len(children)
        metadata["auto_merge_child_scores"] = [child.score for child in children]
        parent_id = key.split(":", 1)[1] if ":" in key else ""
        parent_node = parent_nodes.get(parent_id)
        if parent_node:
            metadata.update(parent_node["metadata"])
            metadata["parent_node_id"] = parent_id
            text = parent_node["text"]
        else:
            text = "\n\n".join(
                child.text.strip() for child in children if str(child.text or "").strip()
            )
        metadata["auto_merge_score"] = max(child.score for child in children)
        score = metadata["auto_merge_score"] + AUTO_MERGE_SCORE_BONUS * math.log1p(
            len(children) - 1
        )
        merged.append(
            RetrievalHit(
                text=text,
                score=score,
                metadata=metadata,
                node_id=representative.node_id,
                category=representative.category,
                collection=representative.collection,
                score_type=representative.score_type,
            )
        )

    merged.sort(key=lambda hit: hit.score, reverse=True)
    return merged[:top_k]


def _load_parent_nodes(parent_ids: list[str]) -> dict[str, dict[str, Any]]:
    """批量读取父节点；数据库不可用时返回空结果，由调用方降级到子块文本。"""
    if not parent_ids:
        return {}
    try:
        import psycopg
        from psycopg import sql

        from retrieval.indexing.index import _pg_connection_strings
        from retrieval.indexing.parent_store import RAG_SCHEMA

        sync_url, _ = _pg_connection_strings()
        nodes: dict[str, dict[str, Any]] = {}
        with psycopg.connect(sync_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT chunk_id, text, metadata
                        FROM {}.parent_chunks
                        WHERE chunk_id = ANY(%s)
                        """
                    ).format(sql.Identifier(RAG_SCHEMA)),
                    (list(dict.fromkeys(parent_ids)),),
                )
                for chunk_id, text, metadata in cur.fetchall():
                    node_text = str(text or "").strip()
                    if node_text:
                        nodes[str(chunk_id)] = {
                            "text": node_text,
                            "metadata": _coerce_metadata(metadata),
                        }
        return nodes
    except Exception as exc:
        logger.warning("parent auto-merge lookup skipped error={}", exc)
        return {}


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
                    score_type="bm25",
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
        if not ranked:
            continue
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
        metadata["score_source"] = "rrf"
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
                score_type="rrf",
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
        metadata["score_source"] = "weighted"
        fused.append(
            RetrievalHit(
                text=hit.text,
                score=score,
                metadata=metadata,
                node_id=hit.node_id,
                category=hit.category,
                collection=hit.collection,
                score_type="weighted",
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


def _milvus_filter_expr(filters: MetadataFilters | None) -> str:
    """将可安全下推的精确字段转换为 Milvus 标量过滤表达式。"""
    filters = filters or {}
    clauses: list[str] = []

    def values(value: Any) -> list[Any]:
        if value in (None, "", [], ()):
            return []
        if isinstance(value, (list, tuple, set)):
            return [item for item in value if item not in (None, "")]
        return [value]

    def quoted(value: Any) -> str:
        return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"

    for field in ("doc_id", "ticker", "issuer"):
        items = values(filters.get(field))
        if items:
            clauses.append(f"{field} in [{', '.join(quoted(item) for item in items)}]")

    years = values(filters.get("year", filters.get("fiscal_year")))
    if years:
        normalized: list[str] = []
        for value in years:
            try:
                normalized.append(str(int(value)))
            except (TypeError, ValueError):
                continue
        if normalized:
            clauses.append(f"year in [{', '.join(normalized)}]")

    return " and ".join(clauses)


def _rerank_text(hit: RetrievalHit) -> str:
    text = str(hit.metadata.get("rerank_text") or "").strip()
    return text or hit.text


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
