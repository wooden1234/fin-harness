"""文本排序客户端：支持 DashScope / 讯飞星辰 MaaS。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.core.config import settings

RerankProvider = Literal["dashscope", "xfyun"]


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
    document: str | None = None


def rerank_enabled() -> bool:
    return bool(settings.RERANK_ENABLED)


def rerank_provider() -> RerankProvider:
    return _provider()


def _provider() -> RerankProvider:
    value = str(settings.RERANK_PROVIDER or "dashscope").strip().lower()
    if value in {"xfyun", "spark", "maas", "iflytek"}:
        return "xfyun"
    return "dashscope"


def _resolve_rerank_credentials() -> tuple[str, str, str]:
    if not settings.RERANK_BASE_URL:
        raise RuntimeError("未配置 RERANK_BASE_URL，无法调用 rerank")

    provider = _provider()
    if settings.RERANK_API_KEY:
        api_key = settings.RERANK_API_KEY
    elif provider == "dashscope":
        api_key = settings.DASHSCOPE_API_KEY or settings.QWEN_API_KEY
        if not api_key:
            raise RuntimeError(
                "未配置 RERANK_API_KEY / DASHSCOPE_API_KEY / QWEN_API_KEY，无法调用 rerank"
            )
    else:
        raise RuntimeError("未配置 RERANK_API_KEY，无法调用讯飞 MaaS rerank")

    return api_key, settings.RERANK_BASE_URL, settings.RERANK_MODEL


def rerank_documents(
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[RerankResult]:
    if not documents:
        return []

    provider = _provider()
    if provider == "xfyun":
        return _rerank_xfyun(query=query, documents=documents, top_n=top_n)
    return _rerank_dashscope(query=query, documents=documents, top_n=top_n)


def _uses_compatible_rerank_api(url: str) -> bool:
    return "compatible-api" in str(url or "").rstrip("/")


def _rerank_dashscope(
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[RerankResult]:
    api_key, url, model = _resolve_rerank_credentials()
    if _uses_compatible_rerank_api(url):
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
    else:
        payload = {
            "model": model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": settings.RERANK_RETURN_DOCUMENTS,
                "top_n": top_n,
            },
        }
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=settings.RERANK_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return _parse_rerank_results(response.json())


def _rerank_xfyun(
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[RerankResult]:
    api_key, url, model = _resolve_rerank_credentials()
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "query": query,
            "documents": documents,
        },
        timeout=settings.RERANK_TIMEOUT_SEC,
    )
    response.raise_for_status()
    parsed = _parse_rerank_results(response.json())
    parsed.sort(key=lambda item: item.score, reverse=True)
    return parsed[:top_n]


def _parse_rerank_results(payload: dict[str, Any]) -> list[RerankResult]:
    results = (
        payload.get("output", {}).get("results")
        or payload.get("output", {}).get("rankings")
        or payload.get("results")
        or []
    )

    parsed: list[RerankResult] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if index is None:
            continue
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        document = item.get("document")
        if isinstance(document, dict):
            document = document.get("text") or document.get("content")
        parsed.append(
            RerankResult(
                index=int(index),
                score=float(score or 0.0),
                document=str(document) if document not in (None, "") else None,
            )
        )
    return parsed
