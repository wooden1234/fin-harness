from __future__ import annotations

import pytest

from agents.finance_agent.pdf_agent.retrieval.multi_query_fuse import (
    _protect_original_top_hits,
    fuse_original_and_rewrite_hits,
)
from retrieval import RetrievalHit


def _hit(node_id: str) -> RetrievalHit:
    return RetrievalHit(
        text=f"text-{node_id}",
        score=1.0,
        metadata={"doc_id": node_id, "chunk_index": 1},
        node_id=node_id,
        category="annual_reports",
    )


def test_protect_original_replaces_rewrite_tail_inside_final_window():
    original = [_hit("o1"), _hit("o2")]
    rewrite = [_hit("r1"), _hit("r2"), _hit("r3")]

    protected, retained = _protect_original_top_hits(
        [*rewrite, *original], original, top_k=3, min_original=2
    )

    assert retained == 2
    assert len({hit.node_id for hit in protected[:3]} & {"o1", "o2"}) == 2
    assert len({hit.node_id for hit in protected}) == len(protected)


@pytest.mark.asyncio
async def test_fusion_keeps_configured_number_of_original_candidates():
    original = [_hit("o1"), _hit("o2")]
    rewrite = [_hit("r1"), _hit("r2"), _hit("r3")]

    class _Retriever:
        async def _arerank_hits(self, query, hits, *, top_k):
            del query, top_k
            by_id = {hit.node_id: hit for hit in hits}
            return [by_id[node_id] for node_id in ("r1", "r2", "r3", "o1", "o2")]

    result = await fuse_original_and_rewrite_hits(
        original_query="原始问题",
        original_hits=original,
        rewrite_hits=rewrite,
        retriever=_Retriever(),
        top_k=3,
        min_original=2,
    )

    assert result.protected_original_count == 2
    assert len({hit.node_id for hit in result.hits} & {"o1", "o2"}) == 2
    assert len(result.hits) == 3
