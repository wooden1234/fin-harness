"""DashScope 文本排序客户端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float
    document: str | None = None


def rerank_enabled() -> bool:
    return bool(settings.RERANK_ENABLED)


def _resolve_rerank_credentials() -> tuple[str, str, str]:
    api_key = settings.DASHSCOPE_API_KEY or settings.QWEN_API_KEY
    if not api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY / QWEN_API_KEY，无法调用 rerank")
    if not settings.RERANK_BASE_URL:
        raise RuntimeError("未配置 RERANK_BASE_URL，无法调用 rerank")
    return api_key, settings.RERANK_BASE_URL, settings.RERANK_MODEL


def rerank_documents(
    *,
    query: str,
    documents: list[str],
    top_n: int,
) -> list[RerankResult]:
    api_key, url, model = _resolve_rerank_credentials()
    if not documents:
        return []

    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": settings.RERANK_RETURN_DOCUMENTS,
                "top_n": top_n,
            },
        },
        timeout=settings.RERANK_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    return _parse_rerank_results(payload)


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
