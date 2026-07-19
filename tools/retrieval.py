"""FAQ/PDF 检索工具，包装现有 retrieval 基础设施。"""

from __future__ import annotations

from retrieval import RetrievalHit, get_faq_retriever, get_pdf_retriever
from retrieval.core.filters import infer_pdf_metadata_filters, merge_filters
from retrieval.retrievers.pdf_kb_router import get_pdf_kb_router
from retrieval.retrievers.query_constraints import parse_query_constraints


def faq_search(query: str, *, top_k: int = 3) -> list[RetrievalHit]:
    retriever = get_faq_retriever(top_k=top_k, similarity_threshold=None)
    return retriever.search(query, top_k=top_k)


def pdf_search(query: str, *, top_k: int = 5) -> list[RetrievalHit]:
    route = get_pdf_kb_router().route(query)
    if not route.supported:
        return []
    categories = list(route.categories) or None
    plan = parse_query_constraints(query, knowledge_bases=categories)
    metadata_filters = merge_filters(
        infer_pdf_metadata_filters("", knowledge_bases=categories),
        plan.filters,
    )
    retriever = get_pdf_retriever(
        top_k=top_k,
        similarity_threshold=None,
        metadata_filters=metadata_filters,
        hybrid=True,
    )
    return retriever.search(query, top_k=top_k, metadata_filters=metadata_filters)
