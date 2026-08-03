"""同花顺问财 SkillHub OpenAPI 工具。

该模块只负责把问财查询封装为受治理的只读 Tool，不在这里解析投资建议，
也不把 API Key 写入返回值、日志或 LangGraph 状态。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import timezone
from pathlib import Path
import re
import secrets
from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_core.tools import BaseTool, tool

from app.core.cache import (
    cache_get,
    cache_set,
    make_record,
    normalize_cache_text,
)
from app.core.config import PROJECT_ROOT, settings
from app.core.redis_keys import redis_keys
from skills.runners.iwencai import installed_skill_version, run_installed_skill
from tools.core.base import ToolSpec
from tools.core.registry import register_tool

_LEGACY_SKILL_ID = "legacy-query2data"
_LEGACY_SKILL_VERSION = "1.0.0"
_SKILL_DESCRIPTION_RE = re.compile(
    r"^description:\s*[\"']?(.+?)[\"']?\s*$",
    re.MULTILINE,
)


def _iwencai_skill_root() -> Path:
    configured = Path(settings.IWENCAI_SKILL_ROOT)
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


def skill_description(skill_id: str, *, fallback: str = "") -> str:
    """读取官方 Skill frontmatter 的 description；缺失时回退 fallback。"""
    path = _iwencai_skill_root() / skill_id / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return fallback
    if not text.startswith("---"):
        return fallback
    end = text.find("\n---", 3)
    if end < 0:
        return fallback
    frontmatter = text[3:end]
    match = _SKILL_DESCRIPTION_RE.search(frontmatter)
    if not match:
        return fallback
    description = match.group(1).strip()
    return description or fallback


def _apply_skill_description(
    langchain_tool: BaseTool,
    skill_id: str,
    *,
    fallback: str,
) -> str:
    """把官方技能描述赋给 LangChain Tool（模型可见）。"""
    description = skill_description(skill_id, fallback=fallback)
    langchain_tool.description = description
    return description

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


_MIN_COMPARE_ENTITIES = 2
_MAX_COMPARE_ENTITIES = 6
_CURRENCY_HINTS: tuple[tuple[str, str], ...] = (
    ("港元", "HKD"),
    ("HKD", "HKD"),
    ("美元", "USD"),
    ("USD", "USD"),
    ("US$", "USD"),
    ("人民币", "CNY"),
    ("CNY", "CNY"),
    ("RMB", "CNY"),
)
_FISCAL_HINT_RE = re.compile(r"(?:FY\s*)?(20\d{2})(?:\s*年)?(?:\s*Q([1-4]))?", re.I)


def _detect_currency_hint(text: str) -> str:
    """从原始问财返回文本里粗粒度识别币种，供多实体对比时提示口径差异。"""
    for marker, currency in _CURRENCY_HINTS:
        if marker in text:
            return currency
    return ""


def _detect_fiscal_hints(text: str, *, limit: int = 3) -> list[str]:
    """粗粒度提取财年/财季标签，仅用于口径差异提示，不作为精确事实。"""
    hints: list[str] = []
    for match in _FISCAL_HINT_RE.finditer(text):
        year, quarter = match.group(1), match.group(2)
        label = f"FY{year} Q{quarter}" if quarter else f"FY{year}"
        if label not in hints:
            hints.append(label)
        if len(hints) >= limit:
            break
    return hints


async def compare_entities_iwencai(
    entities: list[str],
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """并发查询多个实体的同一指标，避免逐个改写重复调用浪费配额。

    对每个实体拼出「实体 + 查询语句」独立请求问财，合并为按实体归集的结果，
    并对返回文本做币种/财年粗粒度扫描，标记是否存在多币种或多财年口径。
    """
    entities = [str(item).strip() for item in entities if str(item).strip()]
    if len(entities) < _MIN_COMPARE_ENTITIES:
        raise ValueError("compare_entities_requires_at_least_two_entities")
    if len(entities) > _MAX_COMPARE_ENTITIES:
        raise ValueError(f"compare_entities_supports_up_to_{_MAX_COMPARE_ENTITIES}")
    if not query.strip():
        raise ValueError("query_must_not_be_empty")

    per_entity_queries = {entity: f"{entity} {query}".strip() for entity in entities}
    results = await asyncio.gather(
        *(fetch_iwencai(text, limit=limit) for text in per_entity_queries.values()),
        return_exceptions=True,
    )

    per_entity: dict[str, Any] = {}
    per_entity_currency: dict[str, str] = {}
    per_entity_periods: dict[str, list[str]] = {}
    failed_entities: list[str] = []
    currencies: list[str] = []
    periods: list[str] = []
    for entity, result in zip(per_entity_queries.keys(), results, strict=True):
        if isinstance(result, BaseException) or not isinstance(result, dict) or not result.get("ok"):
            failed_entities.append(entity)
            continue
        data = result.get("data")
        per_entity[entity] = data
        haystack = str(data)
        currency = _detect_currency_hint(haystack)
        if currency:
            per_entity_currency[entity] = currency
            if currency not in currencies:
                currencies.append(currency)
        entity_periods = _detect_fiscal_hints(haystack)
        if entity_periods:
            per_entity_periods[entity] = entity_periods
        for period in entity_periods:
            if period not in periods:
                periods.append(period)

    if not per_entity:
        return {
            "ok": False,
            "error": "compare_entities_all_failed",
            "failed_entities": failed_entities,
        }
    return {
        "ok": True,
        "provider": "iwencai",
        "query": query,
        "entities": list(per_entity.keys()),
        "failed_entities": failed_entities,
        "per_entity": per_entity,
        "per_entity_currency": per_entity_currency,
        "per_entity_periods": per_entity_periods,
        "calibre": {
            "currencies": currencies,
            "multi_currency": len(currencies) > 1,
            "periods": periods,
            "multi_period": len(periods) > 1,
        },
    }


@tool(parse_docstring=True)
async def compare_entities_with_iwencai(
    entities: list[str],
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """并发查询多个公司/指数的同一财务或行情指标，用于财报或同业对比。

    比逐个调用 query_iwencai 更省配额，且会在返回中标注 calibre.multi_currency
    / calibre.multi_period，提示是否存在币种或财年口径差异。

    Args:
        entities: 2-6 个待对比实体名称，如 ["腾讯", "阿里巴巴"]。
        query: 对每个实体都适用的指标查询语句，如「近两个完整财年营业收入 归母净利润 销售毛利率」。
        limit: 每个实体查询的返回条数上限。
    """
    return await compare_entities_iwencai(entities, query, limit=limit)


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

register_tool(
    ToolSpec(
        tool_id="iwencai.compare_entities",
        name="compare_entities_with_iwencai",
        description="并发查询多个公司/指数的同一指标并标注币种/财年口径差异，用于财报或同业对比",
        risk_level="low",
        read_only=True,
        timeout_seconds=45.0,
    ),
    langchain_tool=compare_entities_with_iwencai,
    handler=compare_entities_with_iwencai.ainvoke,
)

for _tool_id, _skill_id, _fallback, _langchain_tool in (
    (
        "iwencai.market.query",
        "hithink-market-query",
        "查询股票、ETF 和实时行情数据",
        query_iwencai_market,
    ),
    (
        "iwencai.industry.query",
        "hithink-industry-query",
        "查询行业估值、财务、盈利、行情和板块排名",
        query_iwencai_industry,
    ),
    (
        "iwencai.index.query",
        "hithink-zhishu-query",
        "查询主要指数行情和指标",
        query_iwencai_index,
    ),
    (
        "iwencai.rating.query",
        "hithink-insresearch-query",
        "查询研报评级、业绩预测和机构研究数据",
        query_iwencai_rating,
    ),
    (
        "iwencai.announcement.search",
        "announcement-search",
        "搜索上市公司公告和重大事件",
        search_iwencai_announcement,
    ),
    (
        "iwencai.report.search",
        "report-search",
        "搜索券商研报和机构研究报告",
        search_iwencai_report,
    ),
    (
        "iwencai.fund.screen",
        "hithink-fund-selector",
        "筛选公募基金及其基金经理、业绩和持仓",
        screen_iwencai_fund,
    ),
):
    _description = _apply_skill_description(
        _langchain_tool,
        _skill_id,
        fallback=_fallback,
    )
    register_tool(
        ToolSpec(
            tool_id=_tool_id,
            name=_langchain_tool.name,
            description=_description,
            risk_level="low",
            read_only=True,
        ),
        langchain_tool=_langchain_tool,
        handler=_langchain_tool.ainvoke,
    )

_screen_description = _apply_skill_description(
    screen_iwencai,
    "hithink-astock-selector",
    fallback="通过官方问财 Skill 筛选 A 股并返回结构化候选结果",
)
register_tool(
    ToolSpec(
        tool_id="iwencai.screen",
        name="screen_iwencai",
        description=_screen_description,
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
    "skill_description",
]
