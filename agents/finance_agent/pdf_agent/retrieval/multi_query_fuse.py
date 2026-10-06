"""原 query 与 rewrite query 两路命中的 RRF 融合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval import RetrievalHit
from retrieval.retrievers.retriever import (
    _auto_merge_parent_hits,
    _hit_key,
    _rrf_fuse_hits,
)

FUSION_MODE_NONE = "none"
FUSION_MODE_ORIGINAL_REWRITE_RRF = "original_rewrite_rrf"


@dataclass(frozen=True, slots=True)
class MultiQueryFuseResult:
    """两路融合后的命中与可观测计数。"""

    hits: list[RetrievalHit]
    fused_count: int
    reranked_count: int
    merged_count: int
    protected_original_count: int = 0
    fusion_mode: str = FUSION_MODE_ORIGINAL_REWRITE_RRF


def _protect_original_top_hits(
    ranked: list[RetrievalHit],
    original_hits: list[RetrievalHit],
    *,
    top_k: int,
    min_original: int,
) -> tuple[list[RetrievalHit], int]:
    """在最终窗口中至少保留若干原问候选，避免改写候选将其全部挤出。"""
    limit = max(int(top_k), 1)
    required = min(max(int(min_original), 0), len(original_hits), limit)
    if required == 0:
        return list(ranked), 0

    original_by_key = {_hit_key(hit): hit for hit in original_hits}
    original_order = [_hit_key(hit) for hit in original_hits]
    selected = list(ranked[:limit])
    selected_keys = {_hit_key(hit) for hit in selected}
    retained = sum(1 for key in selected_keys if key in original_by_key)
    for key in original_order:
        if retained >= required:
            break
        if key in selected_keys:
            continue
        replacement = original_by_key[key]
        replace_at = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if _hit_key(selected[index]) not in original_by_key
            ),
            None,
        )
        if replace_at is None:
            if len(selected) >= limit:
                break
            selected.append(replacement)
        else:
            selected_keys.discard(_hit_key(selected[replace_at]))
            selected[replace_at] = replacement
        selected_keys.add(key)
        retained += 1

    # 保留窗口之外的候选供父块合并，但避免重复。
    ordered = list(selected)
    seen = {_hit_key(hit) for hit in ordered}
    for hit in [*ranked[limit:], *original_hits]:
        key = _hit_key(hit)
        if key not in seen:
            seen.add(key)
            ordered.append(hit)
    return ordered, retained


async def fuse_original_and_rewrite_hits(
    *,
    original_query: str,
    original_hits: list[RetrievalHit],
    rewrite_hits: list[RetrievalHit],
    retriever: Any,
    top_k: int,
    min_original: int = 2,
) -> MultiQueryFuseResult:
    """RRF 融合后按原问重排，并保护少量原问候选再做父子块合并。"""
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
    protected, protected_count = _protect_original_top_hits(
        reranked,
        list(original_hits or []),
        top_k=max(int(top_k), 1),
        min_original=min_original,
    )
    merged = _auto_merge_parent_hits(protected, top_k=candidate_top_k)
    return MultiQueryFuseResult(
        hits=list(merged[: max(int(top_k), 1)]),
        fused_count=len(fused),
        reranked_count=len(reranked),
        merged_count=len(merged),
        protected_original_count=protected_count,
        fusion_mode=FUSION_MODE_ORIGINAL_REWRITE_RRF,
    )


__all__ = [
    "FUSION_MODE_NONE",
    "FUSION_MODE_ORIGINAL_REWRITE_RRF",
    "MultiQueryFuseResult",
    "_protect_original_top_hits",
    "fuse_original_and_rewrite_hits",
]
