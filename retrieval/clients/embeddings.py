"""W2：Embedding 工厂（OpenAI 兼容接口，支持 DashScope / 讯飞星辰 MaaS）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.cache import (
    cache_delete,
    cache_get,
    cache_set,
    is_finite_number,
    make_record,
    normalize_cache_text,
    record_cache_event,
    with_fill_lock,
)
from app.core.config import settings
from app.core.redis_keys import redis_keys

EmbeddingProvider = Literal["dashscope", "xfyun"]
_QEMB_DOMAIN = "query_embedding"
_QEMB_DATA_TYPE = "query_embedding"


def embedding_provider() -> EmbeddingProvider:
    value = str(settings.EMBEDDING_PROVIDER or "dashscope").strip().lower()
    if value in {"xfyun", "spark", "maas", "iflytek"}:
        return "xfyun"
    return "dashscope"


def _resolve_embedding_credentials() -> tuple[str, str, str]:
    provider = embedding_provider()
    model = settings.EMBEDDING_MODEL

    if settings.EMBEDDING_API_KEY:
        api_key = settings.EMBEDDING_API_KEY
    elif provider == "dashscope":
        api_key = settings.QWEN_API_KEY or settings.DASHSCOPE_API_KEY
        if not api_key:
            raise RuntimeError(
                "未配置 EMBEDDING_API_KEY / QWEN_API_KEY / DASHSCOPE_API_KEY"
            )
    else:
        raise RuntimeError("未配置 EMBEDDING_API_KEY，无法调用讯飞 MaaS embedding")

    if settings.EMBEDDING_BASE_URL:
        api_base = settings.EMBEDDING_BASE_URL.rstrip("/")
    elif provider == "dashscope":
        api_base = settings.QWEN_BASE_URL.rstrip("/")
        if not api_base:
            raise RuntimeError("未配置 EMBEDDING_BASE_URL / QWEN_BASE_URL")
    else:
        raise RuntimeError("未配置 EMBEDDING_BASE_URL，无法调用讯飞 MaaS embedding")

    return api_key, api_base, model


def embedding_batch_size() -> int:
    if settings.EMBEDDING_BATCH_SIZE > 0:
        return settings.EMBEDDING_BATCH_SIZE
    # 讯飞 v2 gRPC 单次请求约 4MB 上限；4096 维 + 长文本需小 batch
    if embedding_provider() == "xfyun":
        return 8
    return 100


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbedding:
    api_key, api_base, model = _resolve_embedding_credentials()
    return OpenAIEmbedding(
        model_name=model,
        api_key=api_key,
        api_base=api_base,
        dimensions=settings.EMBEDDING_DIM,
        embed_batch_size=embedding_batch_size(),
        max_retries=2,
        timeout=60.0,
    )


def _query_embedding_digest(normalized_query: str) -> str:
    _, api_base, model = _resolve_embedding_credentials()
    digest_source = "|".join(
        [
            embedding_provider(),
            api_base,
            model,
            str(settings.EMBEDDING_DIM),
            normalized_query,
        ]
    )
    return redis_keys.digest(digest_source)


def _query_embedding_cache_key(normalized_query: str):
    return redis_keys.build("qemb", "v1", _query_embedding_digest(normalized_query))


def _query_embedding_lock_key(normalized_query: str):
    return redis_keys.build(
        "qemb",
        "lock",
        _query_embedding_digest(normalized_query),
    )


def _validate_embedding_payload(payload: object) -> list[float]:
    if not isinstance(payload, list):
        raise ValueError("embedding_payload_not_list")
    if len(payload) != int(settings.EMBEDDING_DIM):
        raise ValueError("embedding_dim_mismatch")
    values: list[float] = []
    for item in payload:
        if not is_finite_number(item):
            raise ValueError("embedding_non_finite")
        values.append(float(item))
    return values


async def aget_query_embedding_cached(query: str) -> list[float]:
    """异步获取 query embedding；可选 Redis Cache-Aside，默认关闭。"""
    normalized = normalize_cache_text(query)
    enabled = bool(settings.EMBEDDING_CACHE_ENABLED)
    cache_key = _query_embedding_cache_key(normalized)
    lock_key = _query_embedding_lock_key(normalized)

    async def _read_payload() -> list[float] | None:
        cached = await cache_get(
            cache_key,
            domain=_QEMB_DOMAIN,
            data_type=_QEMB_DATA_TYPE,
            enabled=enabled,
        )
        if cached is None or cached.kind != "record":
            return None
        try:
            return _validate_embedding_payload(cached.payload)
        except ValueError:
            record_cache_event(_QEMB_DOMAIN, "deser_fail")
            await cache_delete(cache_key, domain=_QEMB_DOMAIN, enabled=True)
            return None

    async def _load() -> list[float]:
        model = get_embed_model()
        raw = await model.aget_query_embedding(normalized)
        return _validate_embedding_payload(list(raw))

    async def _write(value: list[float]) -> None:
        await cache_set(
            cache_key,
            make_record(data_type=_QEMB_DATA_TYPE, payload=value),
            domain=_QEMB_DOMAIN,
            ttl_seconds=settings.EMBEDDING_CACHE_TTL_SEC,
            enabled=enabled,
        )

    async def _reader_envelope():
        payload = await _read_payload()
        if payload is None:
            return None
        return make_record(data_type=_QEMB_DATA_TYPE, payload=payload)

    value = await with_fill_lock(
        cache_key=cache_key,
        lock_key=lock_key,
        domain=_QEMB_DOMAIN,
        data_type=_QEMB_DATA_TYPE,
        enabled=enabled,
        lock_ttl_seconds=settings.EMBEDDING_CACHE_LOCK_TTL_SEC,
        lock_wait_ms=settings.EMBEDDING_CACHE_LOCK_WAIT_MS,
        loader=_load,
        writer=_write,
        reader=_reader_envelope,
    )
    return list(value)


__all__ = [
    "aget_query_embedding_cached",
    "embedding_batch_size",
    "embedding_provider",
    "get_embed_model",
]
