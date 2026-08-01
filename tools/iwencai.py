"""同花顺问财 SkillHub OpenAPI 工具。

该模块只负责把问财查询封装为受治理的只读 Tool，不在这里解析投资建议，
也不把 API Key 写入返回值、日志或 LangGraph 状态。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import timezone
import secrets
from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_core.tools import tool

from app.core.cache import (
    cache_get,
    cache_set,
    make_record,
    normalize_cache_text,
)
from app.core.config import settings
from app.core.redis_keys import redis_keys
from skills.runners.iwencai import installed_skill_version, run_installed_skill
from tools.base import ToolSpec
from tools.registry import register_tool

_LEGACY_SKILL_ID = "legacy-query2data"
_LEGACY_SKILL_VERSION = "1.0.0"
_IWENCAI_DOMAIN = "iwencai"
_IWENCAI_DATA_TYPE = "iwencai_result"
_MARKET_SKILLS = frozenset(
    {
        "hithink-market-query",
        "hithink-zhishu-query",
        "hithink-industry-query",
        _LEGACY_SKILL_ID,
    }
)
_DOCUMENT_SKILLS = frozenset(
    {
        "announcement-search",
        "report-search",
        "hithink-insresearch-query",
    }
)
_SCREEN_SKILLS = frozenset(
    {
        "hithink-astock-selector",
        "hithink-fund-selector",
    }
)


def _iwencai_ttl_seconds(skill_id: str) -> int:
    if skill_id in _MARKET_SKILLS:
        return int(settings.IWENCAI_CACHE_MARKET_TTL_SEC)
    if skill_id in _DOCUMENT_SKILLS:
        return int(settings.IWENCAI_CACHE_DOCUMENT_TTL_SEC)
    if skill_id in _SCREEN_SKILLS:
        return int(settings.IWENCAI_CACHE_SCREEN_TTL_SEC)
    return int(settings.IWENCAI_CACHE_MARKET_TTL_SEC)


def _iwencai_cache_key(
    *,
    skill_id: str,
    version: str,
    norm_query: str,
    page: int,
    limit: int,
    call_type: str,
):
    digest = redis_keys.digest(
        f"{norm_query}|{page}|{limit}|{call_type}"
    )
    return redis_keys.build("iwencai", skill_id, version, digest)


def _strip_cache_meta(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(payload)
    cleaned.pop("cache_status", None)
    cleaned.pop("cached_at", None)
    return cleaned


async def _run_cached_skill(
    skill_id: str,
    *,
    version: str,
    query: str,
    page: int,
    limit: int,
    call_type: str,
    executor: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """所有问财只读入口的唯一缓存层。"""
    norm_query = normalize_cache_text(query)
    normalized_call_type = str(call_type or "normal").strip().lower() or "normal"
    page_value = int(page)
    limit_value = int(limit)
    enabled = bool(settings.IWENCAI_CACHE_ENABLED)

    async def _execute() -> dict[str, Any]:
        if executor is not None:
            return await executor(
                skill_id=skill_id,
                query=norm_query,
                page=page_value,
                limit=limit_value,
                call_type=normalized_call_type,
                version=version,
            )
        return await run_installed_skill(
            skill_id,
            query=norm_query,
            page=page_value,
            limit=limit_value,
            call_type=normalized_call_type,
        )

    if normalized_call_type == "retry":
        result = await _execute()
        payload = dict(result)
        payload["cache_status"] = "bypass"
        return payload

    cache_key = _iwencai_cache_key(
        skill_id=skill_id,
        version=version,
        norm_query=norm_query,
        page=page_value,
        limit=limit_value,
        call_type=normalized_call_type,
    )
    cached = await cache_get(
        cache_key,
        domain=_IWENCAI_DOMAIN,
        data_type=_IWENCAI_DATA_TYPE,
        enabled=enabled,
    )
    if cached is not None and cached.kind == "record" and isinstance(cached.payload, dict):
        hit = deepcopy(cached.payload)
        hit["cache_status"] = "hit"
        hit["cached_at"] = cached.cached_at.astimezone(timezone.utc).isoformat()
        return hit

    result = await _execute()
    payload = dict(result)
    if payload.get("ok") is True:
        await cache_set(
            cache_key,
            make_record(
                data_type=_IWENCAI_DATA_TYPE,
                payload=_strip_cache_meta(payload),
            ),
            domain=_IWENCAI_DOMAIN,
            ttl_seconds=_iwencai_ttl_seconds(skill_id),
            enabled=enabled,
            max_bytes=int(settings.IWENCAI_CACHE_MAX_BYTES),
        )
        payload["cache_status"] = "miss"
    return payload


async def _run_query_skill(
    skill_id: str,
    query: str,
    *,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """调用已审核的结构化问财 Skill。"""
    return await _run_cached_skill(
        skill_id,
        version=installed_skill_version(skill_id),
        query=query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


async def _run_search_skill(
    skill_id: str,
    query: str,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """调用已审核的问财搜索 Skill。"""
    return await _run_cached_skill(
        skill_id,
        version=installed_skill_version(skill_id),
        query=query,
        page=1,
        limit=limit,
        call_type="normal",
    )


def _base_url() -> str:
    return settings.IWENCAI_BASE_URL.rstrip("/") + "/"


def _api_key() -> str:
    return settings.IWENCAI_API_KEY.strip()


def _validate_limit(limit: int) -> int:
    if limit < 1 or limit > settings.IWENCAI_MAX_LIMIT:
        raise ValueError(f"limit_must_be_between_1_and_{settings.IWENCAI_MAX_LIMIT}")
    return limit


async def _fetch_iwencai_http(
    *,
    skill_id: str,
    query: str,
    page: int,
    limit: int,
    call_type: str,
    version: str,
) -> dict[str, Any]:
    """legacy query2data HTTP 执行器；不含缓存逻辑。"""
    del skill_id, call_type  # 固定协议，仅复用统一签名
    api_key = _api_key()
    if not api_key:
        return {
            "ok": False,
            "error": "iwencai_not_configured",
            "message": "请配置 IWENCAI_API_KEY 后再调用问财工具。",
        }

    payload = {
        "query": query,
        "page": str(page),
        "limit": str(limit),
        "is_cache": "1",
        "expand_index": "true",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Claw-Call-Type": "normal",
        "X-Claw-Skill-Id": _LEGACY_SKILL_ID,
        "X-Claw-Skill-Version": version,
        "X-Claw-Plugin-Id": "none",
        "X-Claw-Plugin-Version": "none",
        "X-Claw-Trace-Id": secrets.token_hex(32),
    }

    try:
        async with httpx.AsyncClient(timeout=settings.IWENCAI_TIMEOUT_SEC) as client:
            response = await client.post(
                urljoin(_base_url(), "v1/query2data"),
                json=payload,
                headers=headers,
            )
    except httpx.TimeoutException:
        return {"ok": False, "error": "iwencai_timeout"}
    except httpx.HTTPError as exc:
        return {"ok": False, "error": "iwencai_network_error", "detail": str(exc)}

    if response.status_code in {401, 403}:
        return {"ok": False, "error": "iwencai_auth_failed"}
    if response.status_code >= 400:
        return {
            "ok": False,
            "error": "iwencai_http_error",
            "status_code": response.status_code,
        }

    try:
        body = response.json()
    except ValueError:
        return {"ok": False, "error": "iwencai_invalid_json"}

    status_code = body.get("status_code")
    if status_code not in (None, 0, "0"):
        return {
            "ok": False,
            "error": "iwencai_api_error",
            "status_code": status_code,
            "message": body.get("status_msg") or body.get("message") or "",
        }

    return {
        "ok": True,
        "provider": "iwencai",
        "query": query,
        "page": page,
        "limit": limit,
        "data": body.get("data", body),
    }


async def fetch_iwencai(
    query: str,
    *,
    page: int = 1,
    limit: int = 10,
) -> dict[str, Any]:
    """调用问财结构化查询接口，返回原始数据和最小元数据。"""
    if not query.strip():
        raise ValueError("query_must_not_be_empty")
    if page < 1:
        raise ValueError("page_must_be_positive")
    normalized_limit = _validate_limit(limit)
    return await _run_cached_skill(
        _LEGACY_SKILL_ID,
        version=_LEGACY_SKILL_VERSION,
        query=query,
        page=page,
        limit=normalized_limit,
        call_type="normal",
        executor=_fetch_iwencai_http,
    )


@tool(parse_docstring=True)
async def query_iwencai(query: str, page: int = 1, limit: int = 10) -> dict[str, Any]:
    """使用自然语言查询同花顺问财数据。

    Args:
        query: 股票、指数、财务或选股查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数，最大值由配置控制。
    """
    return await fetch_iwencai(query, page=page, limit=limit)


@tool(parse_docstring=True)
async def screen_iwencai(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """通过已安装的官方问财 Skill 执行 A 股选股。

    Args:
        query: 自然语言选股条件。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    skill_id = "hithink-astock-selector"
    return await _run_cached_skill(
        skill_id,
        version=installed_skill_version(skill_id),
        query=query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def query_iwencai_market(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """查询股票、ETF 和实时行情数据。

    Args:
        query: 行情查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-market-query",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def query_iwencai_industry(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """查询行业估值、财务、盈利、行情和板块排名。

    Args:
        query: 行业数据查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-industry-query",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def query_iwencai_index(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """查询上证、沪深 300、创业板和海外指数数据。

    Args:
        query: 指数数据查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-zhishu-query",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def query_iwencai_rating(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """查询研报评级、业绩预测、ESG 和机构研究数据。

    Args:
        query: 机构研究或评级查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-insresearch-query",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def screen_iwencai_fund(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """筛选公募基金及其基金经理、业绩和持仓。

    Args:
        query: 基金筛选语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-fund-selector",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


@tool(parse_docstring=True)
async def search_iwencai_announcement(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """搜索上市公司公告和重大事件。

    Args:
        query: 公告搜索语句。
        limit: 返回结果数量。
    """
    return await _run_search_skill("announcement-search", query, limit=limit)


@tool(parse_docstring=True)
async def search_iwencai_report(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """搜索券商研报和机构研究报告。

    Args:
        query: 研报搜索语句。
        limit: 返回结果数量。
    """
    return await _run_search_skill("report-search", query, limit=limit)


register_tool(
    ToolSpec(
        tool_id="iwencai.query",
        name="query_iwencai",
        description="通过同花顺问财自然语言查询股票、指数和选股数据",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=query_iwencai,
    handler=query_iwencai.ainvoke,
)

for _tool_spec, _langchain_tool in (
    (
        ToolSpec(
            tool_id="iwencai.market.query",
            name="query_iwencai_market",
            description="查询股票、ETF 和实时行情数据",
            risk_level="low",
            read_only=True,
        ),
        query_iwencai_market,
    ),
    (
        ToolSpec(
            tool_id="iwencai.industry.query",
            name="query_iwencai_industry",
            description="查询行业估值、财务、盈利、行情和板块排名",
            risk_level="low",
            read_only=True,
        ),
        query_iwencai_industry,
    ),
    (
        ToolSpec(
            tool_id="iwencai.index.query",
            name="query_iwencai_index",
            description="查询主要指数行情和指标",
            risk_level="low",
            read_only=True,
        ),
        query_iwencai_index,
    ),
    (
        ToolSpec(
            tool_id="iwencai.rating.query",
            name="query_iwencai_rating",
            description="查询研报评级、业绩预测和机构研究数据",
            risk_level="low",
            read_only=True,
        ),
        query_iwencai_rating,
    ),
    (
        ToolSpec(
            tool_id="iwencai.announcement.search",
            name="search_iwencai_announcement",
            description="搜索上市公司公告和重大事件",
            risk_level="low",
            read_only=True,
        ),
        search_iwencai_announcement,
    ),
    (
        ToolSpec(
            tool_id="iwencai.report.search",
            name="search_iwencai_report",
            description="搜索券商研报和机构研究报告",
            risk_level="low",
            read_only=True,
        ),
        search_iwencai_report,
    ),
    (
        ToolSpec(
            tool_id="iwencai.fund.screen",
            name="screen_iwencai_fund",
            description="筛选公募基金及其基金经理、业绩和持仓",
            risk_level="low",
            read_only=True,
        ),
        screen_iwencai_fund,
    ),
):
    register_tool(
        _tool_spec,
        langchain_tool=_langchain_tool,
        handler=_langchain_tool.ainvoke,
    )

register_tool(
    ToolSpec(
        tool_id="iwencai.screen",
        name="screen_iwencai",
        description="通过官方问财 Skill 筛选 A 股并返回结构化候选结果",
        risk_level="low",
        read_only=True,
        timeout_seconds=settings.IWENCAI_SKILL_RUNNER_TIMEOUT_SEC,
    ),
    langchain_tool=screen_iwencai,
    handler=screen_iwencai.ainvoke,
)


__all__ = [
    "fetch_iwencai",
    "query_iwencai",
    "query_iwencai_index",
    "query_iwencai_industry",
    "query_iwencai_market",
    "query_iwencai_rating",
    "screen_iwencai",
    "screen_iwencai_fund",
    "search_iwencai_announcement",
    "search_iwencai_report",
]
