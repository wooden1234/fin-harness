"""LLM 工厂：统一管理项目内聊天大模型（DeepSeek / Qwen Vision 等）。

业务代码禁止自行 new Chat* 或直连 chat/completions；一律从本模块取模型。
Embedding / Rerank 属于检索客户端，不在此管理。
"""

from __future__ import annotations

from functools import lru_cache
from time import monotonic

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="llm")

# (checked_at_monotonic, reachable)
_finance_probe_cache: tuple[float, bool] | None = None


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


def _build_openai_compatible_llm(
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    http_async_client: httpx.AsyncClient | None = None,
    extra_body: dict | None = None,
    request_timeout: float | None = None,
) -> ChatOpenAI:
    kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "base_url": _normalize_api_base(base_url),
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


def _finance_llm_reachable(*, base_url: str, api_key: str) -> bool:
    """探活本地 vLLM（GET /models）；带短 TTL，避免每次装配都打满探活。"""
    global _finance_probe_cache
    now = monotonic()
    ttl = max(0.0, float(settings.FINANCE_LLM_PROBE_TTL_SEC or 30.0))
    if _finance_probe_cache is not None:
        checked_at, reachable = _finance_probe_cache
        if now - checked_at < ttl:
            return reachable

    normalized = _normalize_api_base(base_url)
    probe_timeout = max(0.2, float(settings.FINANCE_LLM_PROBE_TIMEOUT_SEC or 2.0))
    reachable = False
    try:
        response = httpx.get(
            f"{normalized}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=probe_timeout,
        )
        reachable = response.status_code < 500
    except Exception as exc:
        logger.warning(
            "finance llm probe failed, fallback to deepseek: base_url={} error={}",
            normalized,
            type(exc).__name__,
        )
    else:
        if not reachable:
            logger.warning(
                "finance llm probe unhealthy, fallback to deepseek: base_url={} status={}",
                normalized,
                response.status_code,
            )
    _finance_probe_cache = (now, reachable)
    return reachable


def reset_finance_llm_probe_cache() -> None:
    """测试或运维手动清掉探活缓存。"""
    global _finance_probe_cache
    _finance_probe_cache = None


@lru_cache(maxsize=1)
def get_router_llm() -> BaseChatModel:
    """Supervisor / 路由 / 结构化抽取：DeepSeek，低温度。"""
    return _build_deepseek_llm(temperature=settings.AGENT_ROUTER_TEMPERATURE)


@lru_cache(maxsize=1)
def get_faq_llm() -> BaseChatModel:
    """对话生成 / FAQ / 通用回答：DeepSeek，略高温度。"""
    return _build_deepseek_llm(temperature=settings.AGENT_FAQ_TEMPERATURE)


@lru_cache(maxsize=1)
def _get_finance_llm_client() -> BaseChatModel:
    """已确认可达时缓存的本地金融微调客户端。"""
    base_url = str(settings.FINANCE_LLM_BASE_URL or "").strip()
    model = str(settings.FINANCE_LLM_MODEL or "").strip()
    api_key = str(settings.FINANCE_LLM_API_KEY or "").strip() or "EMPTY"
    thinking_enabled = bool(settings.FINANCE_LLM_ENABLE_THINKING)
    timeout = max(30.0, float(settings.FINANCE_LLM_TIMEOUT_SEC or 120.0))
    return _build_openai_compatible_llm(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(settings.FINANCE_LLM_TEMPERATURE),
        http_async_client=_get_async_http_client(),
        request_timeout=timeout,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
        },
    )


def get_finance_llm() -> BaseChatModel:
    """DeepAgent 主推理 + 结构化成稿：本地金融微调（OpenAI 兼容）。

    未配置或 vLLM 探活失败时回退 get_faq_llm()（DeepSeek）。
    """
    base_url = str(settings.FINANCE_LLM_BASE_URL or "").strip()
    model = str(settings.FINANCE_LLM_MODEL or "").strip()
    if not base_url or not model:
        return get_faq_llm()

    api_key = str(settings.FINANCE_LLM_API_KEY or "").strip() or "EMPTY"
    if not _finance_llm_reachable(base_url=base_url, api_key=api_key):
        return get_faq_llm()
    return _get_finance_llm_client()


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
    "get_finance_llm",
    "get_pdf_llm",
    "get_qwen_llm",
    "get_router_llm",
    "get_vision_llm",
    "reset_finance_llm_probe_cache",
]
