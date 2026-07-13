"""W2：Embedding 工厂（OpenAI 兼容接口）。"""

from functools import lru_cache

from llama_index.embeddings.openai import OpenAIEmbedding

from app.core.config import settings


def _resolve_embedding_credentials() -> tuple[str, str, str]:
    api_key = settings.QWEN_API_KEY or settings.DASHSCOPE_API_KEY
    api_base = settings.QWEN_BASE_URL
    model = settings.EMBEDDING_MODEL

    if not api_key:
        raise RuntimeError(
            "未配置 Embedding API Key，请在 .env 中设置 QWEN_API_KEY 或 DASHSCOPE_API_KEY"
        )
    if not api_base:
        raise RuntimeError("未配置 QWEN_BASE_URL，请在 .env 中设置 OpenAI 兼容接口地址")
    return api_key, api_base, model


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbedding:
    api_key, api_base, model = _resolve_embedding_credentials()
    return OpenAIEmbedding(
        model_name=model,
        api_key=api_key,
        api_base=api_base,
        dimensions=settings.EMBEDDING_DIM,
        max_retries=2,
        timeout=60.0,
    )
