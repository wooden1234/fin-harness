"""FAQ/PDF 检索工具，包装现有 retrieval 基础设施。"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from retrieval import RetrievalHit, get_faq_retriever, get_pdf_retriever
from retrieval.core.filters import merge_filters
from retrieval.retrievers.pdf_kb_router import get_pdf_kb_router
from retrieval.retrievers.query_filter_extractor import get_query_filter_extractor


def faq_search(query: str, *, top_k: int = 3) -> list[RetrievalHit]:
    retriever = get_faq_retriever(top_k=top_k, similarity_threshold=None)
    return retriever.search(query, top_k=top_k)


def _build_pdf_filters(query: str) -> tuple[dict[str, Any] | None, bool]:
    """LLM 类别路由 + query_filter_llm；返回 (filters, abstained)。"""
    route = get_pdf_kb_router(min_confidence=0.5).route(query)
    if not route.supported:
        return None, True

    categories = list(route.categories)
    filters: dict[str, Any] = {}
    if categories:
        filters["category"] = categories[0] if len(categories) == 1 else categories

    extraction = get_query_filter_extractor().extract(
        query,
        knowledge_bases=categories or None,
    )
    if extraction.filters:
        filters = merge_filters(filters, extraction.filters)
    return (filters or None), False


def pdf_search(query: str, *, top_k: int | None = None) -> list[RetrievalHit]:
    metadata_filters, abstained = _build_pdf_filters(query)
    if abstained:
        return []
    resolved_top_k = max(int(top_k if top_k is not None else settings.PDF_RETRIEVAL_TOP_K), 1)
    retriever = get_pdf_retriever(
        top_k=resolved_top_k,
        similarity_threshold=None,
        metadata_filters=metadata_filters,
        hybrid=True,
        rerank_min_score=settings.RERANK_MIN_SCORE,
    )
    return retriever.search(
        query, top_k=resolved_top_k, metadata_filters=metadata_filters
    )
