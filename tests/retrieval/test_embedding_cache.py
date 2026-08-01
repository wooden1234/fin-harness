"""Query embedding 缓存校验测试。"""

from __future__ import annotations

import math
from types import MethodType

import pytest

from app.core.cache import reset_cache_metrics
from retrieval.clients import embeddings as emb
from retrieval.retrievers import retriever as retriever_module


def test_validate_embedding_payload_rejects_bad_vectors(monkeypatch) -> None:
    monkeypatch.setattr(emb.settings, "EMBEDDING_DIM", 3)
    assert emb._validate_embedding_payload([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="embedding_dim_mismatch"):
        emb._validate_embedding_payload([1.0, 2.0])
    with pytest.raises(ValueError, match="embedding_non_finite"):
        emb._validate_embedding_payload([1.0, math.nan, 3.0])
    with pytest.raises(ValueError, match="embedding_non_finite"):
        emb._validate_embedding_payload([1.0, math.inf, 3.0])


@pytest.mark.asyncio
async def test_aget_query_embedding_cached_hit(monkeypatch) -> None:
    reset_cache_metrics()
    monkeypatch.setattr(emb.settings, "EMBEDDING_CACHE_ENABLED", True)
    monkeypatch.setattr(emb.settings, "EMBEDDING_DIM", 2)
    monkeypatch.setattr(emb, "embedding_provider", lambda: "dashscope")
    monkeypatch.setattr(
        emb,
        "_resolve_embedding_credentials",
        lambda: ("key", "https://example.test", "model-x"),
    )

    calls = {"n": 0}

    class FakeModel:
        async def aget_query_embedding(self, query: str):
            calls["n"] += 1
            return [0.1, 0.2]

    monkeypatch.setattr(emb, "get_embed_model", lambda: FakeModel())

    store: dict[str, str] = {}

    class FakeClient:
        async def get(self, key):
            return store.get(str(key))

        async def set(self, key, value, *, ttl_seconds, only_if_absent=False):
            store[str(key)] = value
            return True

        async def delete(self, key):
            store.pop(str(key), None)
            return 1

        async def acquire_lock(self, key, token, *, ttl_seconds):
            return True

        async def release_lock(self, key, token):
            return True

    client = FakeClient()
    import app.core.cache as cache_mod

    monkeypatch.setattr(cache_mod, "get_redis_client", lambda: client)

    first = await emb.aget_query_embedding_cached("什么是 ROE")
    second = await emb.aget_query_embedding_cached("什么是   ROE")
    assert first == [0.1, 0.2]
    assert second == [0.1, 0.2]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_hybrid_retrieval_caches_only_non_empty_exact_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        retriever_module.settings,
        "RETRIEVAL_EVIDENCE_CACHE_ENABLED",
        True,
    )
    monkeypatch.setattr(retriever_module.settings, "CACHE_TTL_JITTER_RATIO", 0.0)
    store: dict[str, str] = {}

    class FakeClient:
        async def get(self, key):
            return store.get(str(key))

        async def set(self, key, value, *, ttl_seconds, only_if_absent=False):
            store[str(key)] = value
            return True

        async def delete(self, key):
            store.pop(str(key), None)
            return 1

    import app.core.cache as cache_module

    monkeypatch.setattr(cache_module, "get_redis_client", lambda: FakeClient())
    retriever = object.__new__(retriever_module.HybridRetriever)
    retriever.categories = ["faq"]
    retriever.top_k = 3
    retriever.metadata_filters = {}
    retriever.fusion_mode = "rrf"
    retriever.vector_weight = 0.65
    retriever.rrf_k = 60
    retriever.candidate_top_k = 20
    retriever.vector_retriever = type(
        "VectorConfig", (), {"similarity_threshold": 0.35}
    )()
    retriever.rerank_provider = None
    retriever.rerank_model = None
    retriever.rerank_min_score = 0.0
    retriever.last_trace = None
    calls = {"n": 0}

    async def fake_uncached(self, query, top_k=None, metadata_filters=None):
        del self, query, top_k, metadata_filters
        calls["n"] += 1
        return [
            retriever_module.RetrievalHit(
                text="ROE 是净资产收益率。",
                score=0.91,
                metadata={"source": "faq.md", "domain": "capital_market"},
                node_id="faq-1",
                category="faq",
                collection="faq-v1",
                score_type="rerank",
            )
        ]

    retriever._asearch_uncached = MethodType(fake_uncached, retriever)

    first = await retriever.asearch(
        "什么是 ROE",
        metadata_filters={"domain": "capital_market"},
    )
    second = await retriever.asearch(
        "什么是   ROE",
        metadata_filters={"domain": "capital_market"},
    )

    assert first[0].node_id == "faq-1"
    assert second[0].text == first[0].text
    assert calls["n"] == 1
    assert retriever.last_trace.extra["cache_status"] == "hit"


@pytest.mark.asyncio
async def test_hybrid_retrieval_does_not_negative_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        retriever_module.settings,
        "RETRIEVAL_EVIDENCE_CACHE_ENABLED",
        True,
    )
    store: dict[str, str] = {}

    class FakeClient:
        async def get(self, key):
            return store.get(str(key))

        async def set(self, key, value, *, ttl_seconds, only_if_absent=False):
            store[str(key)] = value
            return True

    import app.core.cache as cache_module

    monkeypatch.setattr(cache_module, "get_redis_client", lambda: FakeClient())
    retriever = object.__new__(retriever_module.HybridRetriever)
    retriever.categories = ["faq"]
    retriever.top_k = 3
    retriever.metadata_filters = {}
    retriever.fusion_mode = "rrf"
    retriever.vector_weight = 0.65
    retriever.rrf_k = 60
    retriever.candidate_top_k = 20
    retriever.vector_retriever = type(
        "VectorConfig", (), {"similarity_threshold": 0.35}
    )()
    retriever.rerank_provider = None
    retriever.rerank_model = None
    retriever.rerank_min_score = 0.0
    calls = {"n": 0}

    async def fake_uncached(self, query, top_k=None, metadata_filters=None):
        del self, query, top_k, metadata_filters
        calls["n"] += 1
        return []

    retriever._asearch_uncached = MethodType(fake_uncached, retriever)

    assert await retriever.asearch("不存在的问题") == []
    assert await retriever.asearch("不存在的问题") == []
    assert calls["n"] == 2
    assert store == {}
