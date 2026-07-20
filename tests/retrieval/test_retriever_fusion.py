from retrieval.retrievers.retriever import (
    HybridRetriever,
    RetrievalHit,
    VectorRetriever,
    _auto_merge_parent_hits,
    _rrf_fuse_hits,
    _scale_vector_diversity_penalty,
    _select_vector_hits,
    _weighted_fuse_hits,
)


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


def test_rrf_tolerates_none_ranked_list():
    vector_hits = [_hit("a", 0.9), _hit("b", 0.8)]

    fused = _rrf_fuse_hits(
        [("vector", vector_hits), ("lexical", None)],
        top_k=2,
        rrf_k=60,
    )

    assert [hit.node_id for hit in fused] == ["a", "b"]


def test_vector_search_returns_list_when_enforce_on_empty_false(monkeypatch):
    retriever = VectorRetriever.__new__(VectorRetriever)
    retriever.top_k = 5
    retriever.metadata_filters = {}
    retriever.categories = ["annual_reports"]
    retriever.similarity_threshold = None
    retriever.candidate_multiplier = 1
    retriever.diversify = False
    retriever.last_trace = None
    class FakeClient:
        def has_collection(self, name):
            return True

        def search(self, **kwargs):
            return [[{"id": "a", "distance": 0.1, "entity": {"text": "text-a", "chunk_id": "a"}}]]

    retriever._client = FakeClient()
    retriever._embed_model = type(
        "EmbedModel",
        (),
        {"get_query_embedding": lambda self, query: [0.1, 0.2]},
    )()

    monkeypatch.setattr(retriever, "_ensure_milvus", lambda: True)
    monkeypatch.setattr(
        "retrieval.retrievers.retriever._filtered_categories",
        lambda categories, filters: ["annual_reports"],
    )
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.collection_name",
        lambda category: category,
    )

    hits = retriever.search("q", top_k=1, enforce_on_empty=False)

    assert hits is not None
    assert len(hits) == 1
    assert hits[0].node_id == "a"


def test_rerank_failure_keeps_fusion_score_and_marks_fallback(monkeypatch):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rerank_enabled = True
    retriever.rerank_provider = "test"
    retriever.rerank_model = "test"
    retriever.rerank_candidate_top_k = 20
    hits = [_hit("a", 0.016)]
    hits[0].score_type = "rrf"
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.rerank_documents",
        lambda **_: (_ for _ in ()).throw(TimeoutError("timeout")),
    )

    result = retriever._rerank_hits("q", hits, top_k=1)

    assert result[0].score_type == "rrf"
    assert result[0].metadata["rerank_status"] == "fallback_to_fusion"
    assert result[0].metadata["score_source"] == "rrf"
    assert retriever.last_rerank_status == "fallback_to_fusion"


def test_rerank_success_changes_score_type(monkeypatch):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rerank_enabled = True
    retriever.rerank_provider = "test"
    retriever.rerank_model = "test"
    retriever.rerank_candidate_top_k = 20
    retriever.rerank_min_score = 0.0
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.rerank_documents",
        lambda **_: [type("Result", (), {"index": 0, "score": 0.9})()],
    )

    result = retriever._rerank_hits("q", [_hit("a", 0.016)], top_k=1)

    assert result[0].score_type == "rerank"
    assert result[0].metadata["rerank_status"] == "success"
    assert retriever.last_rerank_status == "success"


def test_rerank_query_gate_returns_empty_when_top_score_is_too_low(monkeypatch):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rerank_enabled = True
    retriever.rerank_provider = "test"
    retriever.rerank_model = "test"
    retriever.rerank_candidate_top_k = 20
    retriever.rerank_min_score = 0.7
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.rerank_documents",
        lambda **_: [type("Result", (), {"index": 0, "score": 0.69})()],
    )

    result = retriever._rerank_hits("q", [_hit("a", 0.016)], top_k=1)

    assert result == []
    assert retriever.last_rerank_status == "below_threshold"


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


def test_vector_selection_keeps_multiple_high_quality_chunks_without_hard_doc_dedup(monkeypatch):
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.settings.VECTOR_DIVERSITY_TARGET_DUPLICATE_RATE",
        0.0,
    )
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.settings.VECTOR_DIVERSITY_STRENGTH",
        0.1,
    )
    hits = [
        _hit("a-1", 1.00),
        _hit("a-2", 0.99),
        _hit("a-3", 0.98),
        _hit("b-1", 0.70),
    ]
    for hit in hits[:3]:
        hit.metadata["doc_id"] = "doc-a"
    hits[3].metadata["doc_id"] = "doc-b"

    selected = _select_vector_hits(hits, top_k=3)

    assert len(selected) == 3
    assert selected[0].node_id == "a-1"
    assert sum(hit.metadata["doc_id"] == "doc-a" for hit in selected) >= 2


def test_vector_selection_deduplicates_same_chunk_id():
    first = _hit("same", 1.0)
    duplicate = _hit("same", 0.9)

    selected = _select_vector_hits([first, duplicate], top_k=2)

    assert [hit.node_id for hit in selected] == ["same"]


def test_auto_merge_groups_parent_chunks_and_keeps_evidence_ids(monkeypatch):
    first = _hit("child-1", 0.80)
    second = _hit("child-2", 0.75)
    other = _hit("child-3", 0.70)
    for hit in (first, second):
        hit.metadata.update({"parent_chunk_id": "parent-1", "root_chunk_id": "root-1"})
    other.metadata.update({"parent_chunk_id": "parent-2", "root_chunk_id": "root-1"})
    monkeypatch.setattr(
        "retrieval.retrievers.retriever._load_parent_nodes",
        lambda parent_ids: {
            parent_id: {"text": f"parent-text-{parent_id}", "metadata": {"chunk_level": "L2"}}
            for parent_id in parent_ids
        },
    )

    merged = _auto_merge_parent_hits([first, second, other], top_k=5)

    assert len(merged) == 2
    assert merged[0].metadata["auto_merged"] is True
    assert merged[0].metadata["evidence_child_ids"] == ["child-1", "child-2"]
    assert merged[0].metadata["child_chunk_ids"] == ["child-1", "child-2"]
    assert merged[0].text == "parent-text-parent-1"
    assert merged[0].metadata["parent_node_id"] == "parent-1"


def test_auto_merge_falls_back_to_root_chunk_id():
    first = _hit("child-1", 0.8)
    second = _hit("child-2", 0.7)
    for hit in (first, second):
        hit.metadata.pop("doc_id", None)
        hit.metadata["root_chunk_id"] = "root-1"

    merged = _auto_merge_parent_hits([first, second], top_k=2)

    assert len(merged) == 1
    assert merged[0].metadata["evidence_child_ids"] == ["child-1", "child-2"]


def test_vector_selection_does_not_trade_high_score_chunks_for_low_score_diversity(monkeypatch):
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.settings.VECTOR_DIVERSITY_TARGET_DUPLICATE_RATE",
        0.0,
    )
    monkeypatch.setattr(
        "retrieval.retrievers.retriever.settings.VECTOR_DIVERSITY_MIN_SCORE_RATIO",
        0.85,
    )
    hits = [
        _hit("a-1", 1.00),
        _hit("a-2", 0.99),
        _hit("a-3", 0.98),
        _hit("b-1", 0.70),
    ]
    for hit in hits[:3]:
        hit.metadata["doc_id"] = "doc-a"
    hits[3].metadata["doc_id"] = "doc-b"

    selected = _select_vector_hits(hits, top_k=3)

    assert [hit.node_id for hit in selected] == ["a-1", "a-2", "a-3"]


def test_vector_diversity_penalty_scales_when_quality_candidates_are_insufficient():
    assert _scale_vector_diversity_penalty(0.2, quality_hit_count=5, top_k=10) == 0.1
    assert _scale_vector_diversity_penalty(0.2, quality_hit_count=10, top_k=10) == 0.2
    assert _scale_vector_diversity_penalty(0.2, quality_hit_count=0, top_k=10) == 0.0
