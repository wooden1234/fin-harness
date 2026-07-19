"""PDF RAG 实验型上下文编排节点。

该节点不改变现有检索图，用于后续对比原查询单路召回与多路上下文编排效果。
"""

from __future__ import annotations

import re
from typing import Any

from retrieval import RetrievalHit, get_pdf_retriever
from retrieval.retrievers.retriever import _auto_merge_parent_hits, _rrf_fuse_hits

from ..state import PdfAgentState
from ..trace import append_trace


def _doc_key(hit: RetrievalHit) -> str:
    metadata = hit.metadata or {}
    return str(metadata.get("doc_id") or metadata.get("source") or hit.category or hit.node_id or "unknown")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower()))


def select_diverse_hits(
    hits: list[RetrievalHit],
    *,
    top_k: int,
    max_per_doc: int = 2,
    mmr_lambda: float = 0.75,
) -> list[RetrievalHit]:
    """用轻量 MMR 兼顾相关性和文档多样性，并限制单文档配额。"""
    if top_k <= 0 or not hits:
        return []
    candidates = [(hit, _tokens(hit.text)) for hit in hits]
    max_score = max((hit.score for hit, _ in candidates), default=0.0)
    min_score = min((hit.score for hit, _ in candidates), default=0.0)
    span = max_score - min_score
    selected: list[tuple[RetrievalHit, set[str]]] = []
    doc_counts: dict[str, int] = {}
    while candidates and len(selected) < top_k:
        available = [
            item
            for item in candidates
            if doc_counts.get(_doc_key(item[0]), 0) < max_per_doc
        ]
        if not available:
            available = candidates

        def mmr_value(item: tuple[RetrievalHit, set[str]]) -> float:
            hit, terms = item
            relevance = 1.0 if span <= 0 else (hit.score - min_score) / span
            redundancy = max(
                (
                    len(terms & previous_terms) / max(len(terms | previous_terms), 1)
                    for _, previous_terms in selected
                ),
                default=0.0,
            )
            return mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy

        best = max(available, key=lambda item: (mmr_value(item), item[0].score))
        candidates.remove(best)
        selected.append(best)
        key = _doc_key(best[0])
        doc_counts[key] = doc_counts.get(key, 0) + 1
    return [hit for hit, _ in selected]


def pack_context(
    hits: list[RetrievalHit],
    *,
    token_budget: int,
) -> tuple[str, list[RetrievalHit]]:
    """按近似 token 预算打包完整片段，不对单个片段做硬截断。"""
    if token_budget <= 0:
        return "", []
    max_chars = token_budget * 4
    parts: list[str] = []
    selected: list[RetrievalHit] = []
    used_chars = 0
    for index, hit in enumerate(hits, start=1):
        metadata = hit.metadata or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page_num") or metadata.get("page")
        section = metadata.get("section_path") or metadata.get("section", "")
        prefix = f"[{index}] source={source}"
        if page is not None:
            prefix += f" page={page}"
        if section:
            prefix += f" section={section}"
        part = f"{prefix}\n{hit.text}".strip()
        extra = len(part) + (2 if parts else 0)
        if used_chars + extra > max_chars:
            continue
        parts.append(part)
        selected.append(hit)
        used_chars += extra
    return "\n\n".join(parts), selected


async def context_pipeline_node(
    state: PdfAgentState,
    *,
    config=None,
    top_k: int = 8,
    candidate_top_k: int = 20,
    token_budget: int = 6000,
    max_per_doc: int = 2,
) -> PdfAgentState:
    """实验型多路召回与上下文编排节点，默认不接入现有生产图。"""
    del config
    query = str(state.get("original_query") or state.get("query") or "").strip()
    rewrite_query = str(state.get("rewrite_query") or "").strip()
    retriever = get_pdf_retriever(top_k=candidate_top_k, similarity_threshold=None, hybrid=True)

    # 先取未 rerank 的两路候选，避免每一路召回都调用一次外部 reranker。
    rerank_enabled = bool(getattr(retriever, "rerank_enabled", False))
    retriever.rerank_enabled = False
    try:
        original_hits = retriever.search(query, top_k=candidate_top_k, metadata_filters=None)
        rewrite_hits: list[RetrievalHit] = []
        if rewrite_query and rewrite_query != query:
            rewrite_hits = retriever.search(rewrite_query, top_k=candidate_top_k, metadata_filters=None)
    finally:
        retriever.rerank_enabled = rerank_enabled

    lists = [("original", original_hits)]
    if rewrite_hits:
        lists.append(("rewrite", rewrite_hits))
    fused = _rrf_fuse_hits(lists, top_k=max(candidate_top_k, top_k * 2))
    reranked = retriever._rerank_hits(query, fused, top_k=max(candidate_top_k, top_k * 2))
    merged = _auto_merge_parent_hits(reranked, top_k=max(candidate_top_k, top_k * 2))
    diversified = select_diverse_hits(merged, top_k=top_k, max_per_doc=max_per_doc)
    context, packed_hits = pack_context(diversified, token_budget=token_budget)

    trace_update = append_trace(
        state,
        "context_pipeline",
        status="ok" if packed_hits else "empty",
        original_hits=len(original_hits),
        rewrite_hits=len(rewrite_hits),
        fused_hits=len(fused),
        reranked_hits=len(reranked),
        merged_hits=len(merged),
        diversified_hits=len(diversified),
        packed_hits=len(packed_hits),
        token_budget=token_budget,
        rerank_enabled=rerank_enabled,
    )
    return {
        "hits": packed_hits,
        "context": context,
        "original_hits": original_hits,
        "rewrite_hits": rewrite_hits,
        "context_pipeline_trace": {
            "original_hits": len(original_hits),
            "rewrite_hits": len(rewrite_hits),
            "fused_hits": len(fused),
            "reranked_hits": len(reranked),
            "merged_hits": len(merged),
            "diversified_hits": len(diversified),
            "packed_hits": len(packed_hits),
            "token_budget": token_budget,
        },
        **trace_update,
    }
