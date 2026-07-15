from retrieval.retrievers.retriever import RetrievalHit, _rrf_fuse_hits, _weighted_fuse_hits


def _hit(node_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        text=f"text-{node_id}",
        score=score,
        metadata={"doc_id": node_id},
        node_id=node_id,
        category="annual_reports",
        collection="annual_reports",
    )


def test_rrf_prefers_items_present_in_both_lists():
    vector_hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    lexical_hits = [_hit("b", 12.0), _hit("d", 10.0), _hit("a", 8.0)]

    fused = _rrf_fuse_hits(
        [("vector", vector_hits), ("lexical", lexical_hits)],
        top_k=3,
        rrf_k=60,
    )

    assert [hit.node_id for hit in fused] == ["b", "a", "d"]
    assert fused[0].metadata["vector_rank"] == 2
    assert fused[0].metadata["bm25_rank"] == 1
    assert fused[1].metadata["vector_rank"] == 1
    assert fused[1].metadata["bm25_rank"] == 3
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"] > fused[2].metadata["rrf_score"]


def test_rrf_single_list_still_ranks_by_position():
    vector_hits = [_hit("a", 0.9), _hit("b", 0.8)]

    fused = _rrf_fuse_hits(
        [("vector", vector_hits), ("lexical", [])],
        top_k=2,
        rrf_k=60,
    )

    assert [hit.node_id for hit in fused] == ["a", "b"]
    assert "bm25_rank" not in fused[0].metadata


def test_weighted_fusion_kept_for_backward_compat():
    vector_hits = [_hit("a", 1.0)]
    lexical_hits = [_hit("b", 1.0)]

    fused = _weighted_fuse_hits(
        vector_hits,
        lexical_hits,
        top_k=2,
        vector_weight=0.65,
    )

    assert {hit.node_id for hit in fused} == {"a", "b"}
    assert fused[0].metadata["hybrid_score"] >= fused[1].metadata["hybrid_score"]
