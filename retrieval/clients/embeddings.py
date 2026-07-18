"""W2：Embedding 工厂（OpenAI 兼容接口，支持 DashScope / 讯飞星辰 MaaS）。"""

from functools import lru_cache
from typing import Literal

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.config import settings

EmbeddingProvider = Literal["dashscope", "xfyun"]


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
