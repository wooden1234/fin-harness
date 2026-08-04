"""LLM 工厂：统一管理项目内聊天大模型（DeepSeek / Qwen Vision 等）。

业务代码禁止自行 new Chat* 或直连 chat/completions；一律从本模块取模型。
Embedding / Rerank 属于检索客户端，不在此管理。
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from app.core.config import settings


def _normalize_api_base(url: str) -> str:
    base = str(url or "").strip().rstrip("/")
    if not base:
        return base
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _require_deepseek_api_key() -> str:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY，请在 .env 中设置（Supervisor / Agent 需要）"
        )
    return settings.DEEPSEEK_API_KEY


def _qwen_api_key() -> str:
    return (
        str(settings.VISION_API_KEY or "").strip()
        or str(settings.DASHSCOPE_API_KEY or "").strip()
        or str(settings.QWEN_API_KEY or "").strip()
    )


def _qwen_chat_base_url() -> str:
    base = (
        str(settings.VISION_BASE_URL or "").strip()
        or str(settings.QWEN_BASE_URL or "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return _normalize_api_base(base)


def _vision_model_name() -> str:
    return str(settings.VISION_MODEL or "qwen3.8-max").strip() or "qwen3.8-max"


def _require_qwen_api_key() -> str:
    key = _qwen_api_key()
    if not key:
        raise RuntimeError(
            "未配置 VISION_API_KEY / DASHSCOPE_API_KEY / QWEN_API_KEY"
        )
    return key


@lru_cache(maxsize=1)
def _get_async_http_client() -> httpx.AsyncClient:
    """DeepSeek 等默认异步客户端。"""
    return httpx.AsyncClient()


@lru_cache(maxsize=1)
def _get_vision_http_client() -> httpx.AsyncClient:
    """Vision 专用客户端：读超时与 VISION_TIMEOUT_SEC 对齐。"""
    read = max(30.0, float(settings.VISION_TIMEOUT_SEC or 120.0))
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=read, write=60.0, pool=15.0)
    )


def _build_deepseek_llm(*, temperature: float) -> ChatDeepSeek:
    thinking_type = "enabled" if settings.DEEPSEEK_THINKING_ENABLED else "disabled"
    return ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=_require_deepseek_api_key(),
        api_base=_normalize_api_base(settings.DEEPSEEK_BASE_URL),
        temperature=temperature,
        max_retries=0,
        http_async_client=_get_async_http_client(),
        extra_body={"thinking": {"type": thinking_type}},
    )


def _build_qwen_chat_llm(
    *,
    model: str,
    temperature: float,
    http_async_client: httpx.AsyncClient | None = None,
    extra_body: dict | None = None,
    request_timeout: float | None = None,
) -> ChatOpenAI:
    kwargs: dict = {
        "model": model,
        "api_key": _require_qwen_api_key(),
        "base_url": _qwen_chat_base_url(),
        "temperature": temperature,
        "max_retries": 0,
    }
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    if extra_body:
        kwargs["extra_body"] = extra_body
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_router_llm() -> BaseChatModel:
    """Supervisor / 路由 / 结构化抽取：DeepSeek，低温度。"""
    return _build_deepseek_llm(temperature=settings.AGENT_ROUTER_TEMPERATURE)


@lru_cache(maxsize=1)
def get_faq_llm() -> BaseChatModel:
    """对话生成 / DeepAgent 主模型：DeepSeek，略高温度。"""
    return _build_deepseek_llm(temperature=settings.AGENT_FAQ_TEMPERATURE)


def get_pdf_llm() -> BaseChatModel:
    """PDF 文档回答：复用问答生成模型。"""
    return get_faq_llm()


@lru_cache(maxsize=1)
def get_vision_llm() -> BaseChatModel:
    """多模态看图预处理：Qwen（DashScope OpenAI 兼容）。"""
    read = max(30.0, float(settings.VISION_TIMEOUT_SEC or 120.0))
    return _build_qwen_chat_llm(
        model=_vision_model_name(),
        temperature=0.0,
        http_async_client=_get_vision_http_client(),
        request_timeout=read,
        # 关闭深度思考，缩短看图耗时
        extra_body={"enable_thinking": False},
    )


@lru_cache(maxsize=1)
def get_qwen_llm() -> BaseChatModel:
    """通用 Qwen 文本聊天（与 Vision 同凭证；模型优先 VISION_MODEL，否则回退）。"""
    return _build_qwen_chat_llm(
        model=_vision_model_name(),
        temperature=settings.AGENT_ROUTER_TEMPERATURE,
        http_async_client=_get_async_http_client(),
    )


__all__ = [
    "get_faq_llm",
    "get_pdf_llm",
    "get_qwen_llm",
    "get_router_llm",
    "get_vision_llm",
]
