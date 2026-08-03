"""今日热榜：打开页面时按需搜索财经热点，归入三区问题。

类似微博热搜：不跑定时任务，请求时取「今天」的热榜。
采用 Stale-While-Revalidate：命中当日缓存直接返回；未命中时立即返回昨日缓存
或本地兜底，同时在后台异步重新搜索并调用 LLM 刷新，避免请求被慢路径卡住。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from tools.web_search import fetch_web_search
from agents.llm import get_router_llm
from agents.structured_output import ainvoke_json_output
from app.core.cache import (
    cache_fill_lock,
    cache_get,
    cache_release_fill_lock,
    cache_set,
    make_record,
)
from app.core.logger import get_logger
from app.core.redis_client import get_redis_client
from app.core.redis_keys import RedisKey, redis_keys
from app.schemas.hot_board import HotBoardPanel, HotBoardResponse

logger = get_logger(service="hot_board")

_TZ = ZoneInfo("Asia/Shanghai")
_CACHE_DOMAIN = "hot_board"
_CACHE_DATA_TYPE = "hot_board_panels"
# 缓存 key 不带日期：始终保存「最近一次成功生成」的热榜，靠 as_of 字段判断是否为今日新鲜数据；
# TTL 拉长到超过一天，保证跨天时仍有昨日数据可作为 stale 兜底，而不是直接掉回本地静态文案。
_CACHE_TTL_SEC = 26 * 3600
_REFRESH_LOCK_TTL_SEC = 30
_background_refresh_tasks: set[asyncio.Task[None]] = set()
_PANEL_SPECS: tuple[tuple[str, str], ...] = (
    ("hot_discuss", "热门讨论"),
    ("finance_lookup", "财务查数"),
    ("market_view", "市场研判"),
)
_SEARCH_QUERIES = (
    "今日财经热搜 A股热点话题",
    "今日热门上市公司 财报 营收 净利润",
    "今日热门板块 概念股 市场情绪 估值",
)
_FALLBACK_PANELS: list[dict[str, Any]] = [
    {
        "id": "hot_discuss",
        "title": "热门讨论",
        "items": [
            "信用卡年费怎么收？",
            "什么是 T+1 交易制度？",
            "如何筛选高股息 A 股？",
        ],
    },
    {
        "id": "finance_lookup",
        "title": "财务查数",
        "items": [
            "腾讯 2024 年营业收入是多少？",
            "贵州茅台近五年净利润怎么看？",
            "宁德时代毛利率是多少？",
        ],
    },
    {
        "id": "market_view",
        "title": "市场研判",
        "items": [
            "今天市场情绪怎么样？",
            "算力概念股近期表现如何？",
            "白酒板块估值贵不贵？",
        ],
    },
]


class _HotBoardLLMOutput(BaseModel):
    hot_discuss: list[str] = Field(default_factory=list, max_length=5)
    finance_lookup: list[str] = Field(default_factory=list, max_length=5)
    market_view: list[str] = Field(default_factory=list, max_length=5)


def _today() -> str:
    return datetime.now(_TZ).date().isoformat()


def _fallback(as_of: str) -> HotBoardResponse:
    return HotBoardResponse(
        as_of=as_of,
        source="fallback",
        panels=[HotBoardPanel.model_validate(item) for item in _FALLBACK_PANELS],
    )


def _normalize_items(items: list[str], *, limit: int = 3) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip().rstrip("。.")
        if not text:
            continue
        if not text.endswith("？") and not text.endswith("?"):
            text = f"{text}？"
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _panels_from_llm(
    draft: _HotBoardLLMOutput,
    *,
    as_of: str,
    source: Literal["web", "fallback"],
) -> HotBoardResponse:
    fallback_by_id = {item["id"]: item["items"] for item in _FALLBACK_PANELS}
    panels: list[HotBoardPanel] = []
    for panel_id, title in _PANEL_SPECS:
        items = _normalize_items(list(getattr(draft, panel_id, []) or []))
        if len(items) < 3:
            items = _normalize_items([*items, *fallback_by_id[panel_id]])
        panels.append(HotBoardPanel(id=panel_id, title=title, items=items[:3]))
    return HotBoardResponse(as_of=as_of, source=source, panels=panels)


async def _collect_hot_snippets() -> list[str]:
    snippets: list[str] = []
    results = await asyncio.gather(
        *(fetch_web_search(query, scope="open") for query in _SEARCH_QUERIES),
        return_exceptions=True,
    )
    for query, result in zip(_SEARCH_QUERIES, results, strict=True):
        if isinstance(result, Exception):
            logger.warning("hot board search failed query={} err={}", query, type(result).__name__)
            continue
        if not result.get("configured"):
            continue
        answer = str(result.get("answer") or "").strip()
        if answer:
            snippets.append(f"摘要：{answer}")
        for item in list(result.get("results") or [])[:4]:
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()[:180]
            if title or content:
                snippets.append(f"{title} | {content}".strip(" |"))
    return snippets


async def _generate_from_web(as_of: str) -> HotBoardResponse | None:
    snippets = await _collect_hot_snippets()
    if not snippets:
        return None
    evidence = "\n".join(f"- {item}" for item in snippets[:24])
    prompt = f"""今天是 {as_of}（Asia/Shanghai）。
你是财经产品的「今日热榜」编辑。根据下面检索到的今日热点材料，生成首页三区可点击提问。

要求：
1. hot_discuss：规则、概念、怎么做、产品办理等讨论型问题，3 条
2. finance_lookup：具体公司 + 财务指标/年份的查数问题，3 条；公司名尽量来自热点
3. market_view：板块、概念股、估值、市场情绪、近期表现，3 条
4. 每条 8～28 个汉字，口语、可直接发给投研助手，以中文问号结尾
5. 必须紧贴今日材料，不要编造未出现的冷门公司；材料不足时用常见龙头补足
6. 不要输出投资建议或保证收益语气

今日热点材料：
{evidence}
"""
    draft = await ainvoke_json_output(
        get_router_llm(),
        _HotBoardLLMOutput,
        [
            (
                "system",
                "你只输出合法 JSON，字段为 hot_discuss / finance_lookup / market_view。"
                "每个字段是字符串数组。",
            ),
            ("human", prompt),
        ],
    )
    return _panels_from_llm(draft, as_of=as_of, source="web")


async def _refresh_and_store(as_of: str, cache_key: RedisKey) -> HotBoardResponse:
    """实际执行搜索 + LLM 生成，并把成功结果写回缓存；供同步刷新和后台任务共用。"""
    try:
        board = await _generate_from_web(as_of)
    except Exception:
        logger.exception("hot board generation failed as_of={}", as_of)
        board = None

    if board is None:
        return _fallback(as_of)

    await cache_set(
        cache_key,
        make_record(data_type=_CACHE_DATA_TYPE, payload=board.model_dump(mode="json")),
        domain=_CACHE_DOMAIN,
        ttl_seconds=_CACHE_TTL_SEC,
        enabled=True,
    )
    return board


async def _run_background_refresh(
    as_of: str,
    cache_key: RedisKey,
    lock_key: RedisKey,
    lock_token: str | None,
) -> None:
    try:
        await _refresh_and_store(as_of, cache_key)
    except Exception:
        logger.exception("hot board background refresh failed as_of={}", as_of)
    finally:
        if lock_token is not None:
            await cache_release_fill_lock(lock_key, lock_token, domain=_CACHE_DOMAIN)


async def _trigger_background_refresh(as_of: str, cache_key: RedisKey) -> None:
    """异步刷新今日热榜；有 Redis 时用短租约锁避免并发请求重复触发。"""
    lock_key = redis_keys.build(_CACHE_DOMAIN, "refresh-lock", as_of)
    lock_token: str | None = None
    if get_redis_client() is not None:
        lock_token = await cache_fill_lock(
            lock_key,
            domain=_CACHE_DOMAIN,
            ttl_seconds=_REFRESH_LOCK_TTL_SEC,
            enabled=True,
        )
        if lock_token is None:
            # 已有其它请求在后台刷新，本次直接跳过，等它写回缓存。
            return
    task = asyncio.create_task(
        _run_background_refresh(as_of, cache_key, lock_key, lock_token)
    )
    _background_refresh_tasks.add(task)
    task.add_done_callback(_background_refresh_tasks.discard)


async def get_hot_board(*, force_refresh: bool = False) -> HotBoardResponse:
    """返回今日热榜。

    - `force_refresh=True`（用户显式点刷新）：同步重新生成，保证拿到最新结果。
    - 默认路径（首页打开即触发）：命中当日缓存直接返回；未命中则立即返回昨日缓存
      或本地兜底文案，同时把真正耗时的搜索 + LLM 生成丢到后台任务，不阻塞本次请求。
    """
    as_of = _today()
    cache_key = redis_keys.build(_CACHE_DOMAIN, "latest")

    if force_refresh:
        return await _refresh_and_store(as_of, cache_key)

    cached = await cache_get(
        cache_key,
        domain=_CACHE_DOMAIN,
        data_type=_CACHE_DATA_TYPE,
        enabled=True,
    )
    board: HotBoardResponse | None = None
    if cached is not None and cached.kind == "record":
        try:
            board = HotBoardResponse.model_validate(cached.payload)
        except Exception:
            logger.warning("hot board cache payload invalid as_of={}", as_of)

    if board is not None and board.as_of == as_of:
        return board

    await _trigger_background_refresh(as_of, cache_key)
    if board is not None:
        return board.model_copy(update={"source": "stale"})
    return _fallback(as_of)


__all__ = ["get_hot_board"]
