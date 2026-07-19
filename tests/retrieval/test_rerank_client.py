from retrieval.clients.rerank_client import (
    RerankResult,
    _parse_rerank_results,
    _rerank_xfyun,
    rerank_provider,
)


def test_rerank_provider_xfyun(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.rerank_client.settings.RERANK_PROVIDER",
        "xfyun",
    )
    assert rerank_provider() == "xfyun"


def test_rerank_xfyun_sorts_and_limits_top_n(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            }

    monkeypatch.setattr(
        "retrieval.clients.rerank_client._resolve_rerank_credentials",
        lambda: ("key", "http://example/rerank", "model"),
    )
    monkeypatch.setattr(
        "retrieval.clients.rerank_client.httpx.post",
        lambda *args, **kwargs: _Resp(),
    )

    parsed = _rerank_xfyun(
        query="q",
        documents=["a", "b", "c"],
        top_n=2,
    )
    assert parsed == [
        RerankResult(index=1, score=0.9, document=None),
        RerankResult(index=2, score=0.5, document=None),
    ]


def test_parse_xfyun_rerank_payload() -> None:
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.27},
            {"index": 2, "relevance_score": 0.001},
            {"index": 1, "relevance_score": 0.0002},
        ]
    }
    parsed = _parse_rerank_results(payload)
    assert parsed == [
        RerankResult(index=0, score=0.27, document=None),
        RerankResult(index=2, score=0.001, document=None),
        RerankResult(index=1, score=0.0002, document=None),
    ]


def test_rerank_dashscope_compatible_api(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }

    captured: dict = {}

    def _post(url: str, *, headers: dict, json: dict, timeout: float):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(
        "retrieval.clients.rerank_client._resolve_rerank_credentials",
        lambda: (
            "key",
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            "qwen3-rerank",
        ),
    )
    monkeypatch.setattr("retrieval.clients.rerank_client.httpx.post", _post)

    from retrieval.clients.rerank_client import _rerank_dashscope

    parsed = _rerank_dashscope(query="q", documents=["a", "b"], top_n=2)
    assert captured["json"] == {
        "model": "qwen3-rerank",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
    }
    assert parsed == [
        RerankResult(index=1, score=0.9, document=None),
        RerankResult(index=0, score=0.1, document=None),
    ]


def test_parse_dashscope_rerank_payload() -> None:
    payload = {
        "output": {
            "results": [
                {
                    "index": 1,
                    "relevance_score": 0.9,
                    "document": {"text": "doc-b"},
                }
            ]
        }
    }
    parsed = _parse_rerank_results(payload)
    assert len(parsed) == 1
    assert parsed[0].index == 1
    assert parsed[0].score == 0.9
    assert parsed[0].document == "doc-b"
