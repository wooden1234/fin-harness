"""FAQ/PDF 检索工具，包装现有 retrieval 基础设施。"""

from __future__ import annotations

from retrieval import RetrievalHit, get_faq_retriever, get_pdf_retriever
from retrieval.filters import infer_pdf_metadata_filters
from retrieval.pdf_kb_router import get_pdf_kb_router


def faq_search(query: str, *, top_k: int = 3) -> list[RetrievalHit]:
    retriever = get_faq_retriever(top_k=top_k, similarity_threshold=None)
    return retriever.search(query, top_k=top_k)


def pdf_search(query: str, *, top_k: int = 5) -> list[RetrievalHit]:
    categories = get_pdf_kb_router().route_categories(query) or None
    metadata_filters = infer_pdf_metadata_filters(query, knowledge_bases=categories)
    retriever = get_pdf_retriever(
        top_k=top_k,
        similarity_threshold=None,
        metadata_filters=metadata_filters,
        hybrid=True,
    )
    return retriever.search(query, top_k=top_k, metadata_filters=metadata_filters)
