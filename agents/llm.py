"""LLM 工厂：DeepSeek（langchain-deepseek，Week 3+）。"""

from functools import lru_cache

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from app.core.config import settings


def _normalize_api_base(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _require_llm_api_key() -> str:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY，请在 .env 中设置（Supervisor / Agent 需要）"
        )
    return settings.DEEPSEEK_API_KEY


@lru_cache(maxsize=1)
def _get_async_http_client() -> httpx.AsyncClient:
    """复用异步 HTTP 客户端，避免 LLM 请求回退到同步网络调用。"""
    return httpx.AsyncClient()


def _build_deepseek_llm(*, temperature: float) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=_require_llm_api_key(),
        api_base=_normalize_api_base(settings.DEEPSEEK_BASE_URL),
        temperature=temperature,
        max_retries=2,
        http_async_client=_get_async_http_client(),
    )


@lru_cache(maxsize=1)
def get_router_llm() -> BaseChatModel:
    """Supervisor 路由用 LLM：低温度、结构化输出友好。"""
    return _build_deepseek_llm(temperature=settings.AGENT_ROUTER_TEMPERATURE)


@lru_cache(maxsize=1)
def get_faq_llm() -> BaseChatModel:
    """FAQ 回答用 LLM：略高温度，自然语言生成。"""
    return _build_deepseek_llm(temperature=settings.AGENT_FAQ_TEMPERATURE)


def get_pdf_llm() -> BaseChatModel:
    """PDF 文档回答用 LLM：复用问答生成模型配置。"""
    return get_faq_llm()
