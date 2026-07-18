import pytest

from retrieval.clients.embeddings import (
    _resolve_embedding_credentials,
    embedding_batch_size,
    embedding_provider,
)


def test_embedding_provider_xfyun(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
        "xfyun",
    )
    assert embedding_provider() == "xfyun"


def test_embedding_provider_aliases(monkeypatch) -> None:
    for alias in ("spark", "maas", "iflytek"):
        monkeypatch.setattr(
            "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
            alias,
        )
        assert embedding_provider() == "xfyun"


def test_resolve_xfyun_embedding_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
        "xfyun",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_API_KEY",
        "app:secret",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_BASE_URL",
        "https://maas-api.cn-huabei-1.xf-yun.com/v1",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_MODEL",
        "xop3qwen8bembedding",
    )

    api_key, api_base, model = _resolve_embedding_credentials()
    assert api_key == "app:secret"
    assert api_base == "https://maas-api.cn-huabei-1.xf-yun.com/v1"
    assert model == "xop3qwen8bembedding"


def test_resolve_xfyun_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
        "xfyun",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_API_KEY",
        "",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_BASE_URL",
        "https://maas-api.cn-huabei-1.xf-yun.com/v1",
    )

    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        _resolve_embedding_credentials()


def test_embedding_batch_size_xfyun_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
        "xfyun",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_BATCH_SIZE",
        0,
    )
    assert embedding_batch_size() == 8


def test_embedding_batch_size_env_override(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_BATCH_SIZE",
        16,
    )
    assert embedding_batch_size() == 16

    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_PROVIDER",
        "dashscope",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_API_KEY",
        "",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_BASE_URL",
        "",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.QWEN_API_KEY",
        "sk-qwen",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.DASHSCOPE_API_KEY",
        "",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setattr(
        "retrieval.clients.embeddings.settings.EMBEDDING_MODEL",
        "text-embedding-v2",
    )

    api_key, api_base, model = _resolve_embedding_credentials()
    assert api_key == "sk-qwen"
    assert api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model == "text-embedding-v2"
