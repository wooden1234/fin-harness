"""原 query 与 rewrite query 两路命中的 RRF 融合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval import RetrievalHit
from retrieval.retrievers.retriever import _auto_merge_parent_hits, _rrf_fuse_hits

FUSION_MODE_NONE = "none"
FUSION_MODE_ORIGINAL_REWRITE_RRF = "original_rewrite_rrf"


@dataclass(frozen=True, slots=True)
class MultiQueryFuseResult:
    """两路融合后的命中与可观测计数。"""

    hits: list[RetrievalHit]
    fused_count: int
    reranked_count: int
    merged_count: int
    fusion_mode: str = FUSION_MODE_ORIGINAL_REWRITE_RRF


async def fuse_original_and_rewrite_hits(
    *,
    original_query: str,
    original_hits: list[RetrievalHit],
    rewrite_hits: list[RetrievalHit],
    retriever: Any,
    top_k: int,
) -> MultiQueryFuseResult:
    """RRF 融合两路候选，再按用户原问 Rerank，最后做父子块合并。"""
    candidate_top_k = max(int(top_k) * 2, int(top_k), 1)
    lists: list[tuple[str, list[RetrievalHit]]] = [("original", list(original_hits or []))]
    if rewrite_hits:
        lists.append(("rewrite", list(rewrite_hits)))
    fused = _rrf_fuse_hits(lists, top_k=candidate_top_k)
    reranked = await retriever._arerank_hits(
        original_query,
        fused,
        top_k=candidate_top_k,
    )
    merged = _auto_merge_parent_hits(reranked, top_k=candidate_top_k)
    return MultiQueryFuseResult(
        hits=list(merged[: max(int(top_k), 1)]),
        fused_count=len(fused),
        reranked_count=len(reranked),
        merged_count=len(merged),
        fusion_mode=FUSION_MODE_ORIGINAL_REWRITE_RRF,
    )


__all__ = [
    "FUSION_MODE_NONE",
    "FUSION_MODE_ORIGINAL_REWRITE_RRF",
    "MultiQueryFuseResult",
    "fuse_original_and_rewrite_hits",
]
