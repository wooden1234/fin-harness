"""联网搜索工具：模型只传问句；范围仅 allowlist / open 两种。"""

from __future__ import annotations

from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict

import httpx
from langchain_core.tools import tool

from app.core.config import settings
from app.core.logger import get_logger
from tools.core.base import ToolSpec
from tools.core.registry import register_tool
from tools.web_domains import resolve_search_domains

logger = get_logger(service="web_search_tool")

WebSearchScope = Literal["allowlist", "open"]


class WebSearchResult(TypedDict):
    title: NotRequired[str]
    url: NotRequired[str]
    content: NotRequired[str]
    published_date: NotRequired[str]
    score: NotRequired[float]


class WebSearchScoreStats(TypedDict):
    min: float
    max: float
    mean: float
    raw_count: int
    kept_count: int
    dropped_by_score: int
    fallback_kept: bool


class WebSearchResponse(TypedDict):
    answer: str
    results: list[WebSearchResult]
    configured: bool
    include_domains: NotRequired[list[str]]
    score_stats: NotRequired[WebSearchScoreStats]


def _empty_response(*, configured: bool) -> WebSearchResponse:
    return {
        "answer": "",
        "results": [],
        "configured": configured,
        "include_domains": [],
        "score_stats": {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "raw_count": 0,
            "kept_count": 0,
            "dropped_by_score": 0,
            "fallback_kept": False,
        },
    }


def _rank_and_filter_results(
    results: list[WebSearchResult],
    *,
    min_score: float,
    max_results: int,
) -> tuple[list[WebSearchResult], WebSearchScoreStats]:
    """按 score 降序软过滤；全低于阈值时保底保留最高分 1 条。"""
    ordered = sorted(
        results,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    scores = [float(item.get("score") or 0.0) for item in ordered]
    raw_count = len(ordered)
    if not ordered:
        return [], {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "raw_count": 0,
            "kept_count": 0,
            "dropped_by_score": 0,
            "fallback_kept": False,
        }

    above_threshold = [
        item for item in ordered if float(item.get("score") or 0.0) >= min_score
    ]
    fallback_kept = False
    if above_threshold:
        kept = above_threshold[: max(1, max_results)]
        dropped_by_score = raw_count - len(above_threshold)
    else:
        kept = ordered[:1]
        fallback_kept = True
        dropped_by_score = raw_count - 1
    stats: WebSearchScoreStats = {
        "min": min(scores),
        "max": max(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "raw_count": raw_count,
        "kept_count": len(kept),
        "dropped_by_score": dropped_by_score,
        "fallback_kept": fallback_kept,
    }
    return kept, stats


async def _search_tavily(
    query: str,
    *,
    include_domains: list[str] | None = None,
) -> WebSearchResponse:
    if not settings.TAVILY_API_KEY:
        return _empty_response(configured=False)

    max_results = max(1, min(int(settings.WEB_SEARCH_MAX_RESULTS), 10))
    min_score = float(getattr(settings, "WEB_SEARCH_MIN_SCORE", 0.2) or 0.0)
    # 多取少量候选，软过滤后仍尽量填满 max_results。
    fetch_count = max(max_results, min(10, max_results + 2))
    payload: dict[str, Any] = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": fetch_count,
        "include_answer": True,
        "include_raw_content": False,
    }
    domains = [
        str(item).strip().lower()
        for item in list(include_domains or [])[:5]
        if str(item).strip()
    ]
    if domains:
        payload["include_domains"] = domains

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(settings.TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    results = [
        WebSearchResult(
            title=str(item.get("title") or ""),
            url=str(item.get("url") or ""),
            content=str(item.get("content") or ""),
            published_date=str(item.get("published_date") or ""),
            score=float(item.get("score") or 0),
        )
        for item in list(data.get("results") or [])
        if isinstance(item, dict)
    ]
    kept, score_stats = _rank_and_filter_results(
        results,
        min_score=min_score,
        max_results=max_results,
    )
    logger.info(
        "web_search score_filter query={} raw={} kept={} dropped={} "
        "min={:.3f} max={:.3f} mean={:.3f} fallback={}",
        (query or "")[:80],
        score_stats["raw_count"],
        score_stats["kept_count"],
        score_stats["dropped_by_score"],
        score_stats["min"],
        score_stats["max"],
        score_stats["mean"],
        score_stats["fallback_kept"],
    )
    answer = str(data.get("answer") or "").strip()
    return {
        "answer": answer,
        "results": kept,
        "configured": True,
        "include_domains": domains,
        "score_stats": score_stats,
    }


async def fetch_web_search(
    query: str,
    *,
    scope: WebSearchScope = "allowlist",
) -> WebSearchResponse:
    """执行联网搜索。

    - scope=\"allowlist\"（默认）：金融白名单，最多 5 个域名
    - scope=\"open\"：不限域名（如热榜）
    """
    provider = str(settings.WEB_SEARCH_PROVIDER or "").strip().lower()
    if provider != "tavily":
        logger.warning("unsupported web search provider={}", settings.WEB_SEARCH_PROVIDER)
        return _empty_response(configured=False)

    mode: WebSearchScope = "open" if str(scope).strip().lower() == "open" else "allowlist"
    domains = resolve_search_domains(
        scope=mode,
        allowed_domains=str(getattr(settings, "WEB_SEARCH_ALLOWED_DOMAINS", "") or ""),
        max_domains=5,
    )
    return await _search_tavily(query, include_domains=domains or None)


@tool(parse_docstring=True)
async def search_web(query: str) -> dict[str, Any]:
    """联网搜索公开网页信息，用于最新动态、公开资料补充。

    Args:
        query: 搜索关键词或完整问句
    """
    # 模型入口固定走白名单，避免被配置误开成全网搜。
    return await fetch_web_search(query, scope="allowlist")


register_tool(
    ToolSpec(
        tool_id="web.search",
        name="search_web",
        description="联网搜索公开网页信息；默认限定金融可信域名",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=search_web,
)


__all__ = [
    "WebSearchResponse",
    "WebSearchResult",
    "WebSearchScope",
    "WebSearchScoreStats",
    "_rank_and_filter_results",
    "fetch_web_search",
    "search_web",
]
