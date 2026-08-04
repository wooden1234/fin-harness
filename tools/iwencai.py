"""同花顺问财 SkillHub OpenAPI 工具。

该模块只负责把问财查询封装为受治理的只读 Tool，不在这里解析投资建议，
也不把 API Key 写入返回值、日志或 LangGraph 状态。
"""

from __future__ import annotations

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
    """读取 `.iwencai-skills/<id>/SKILL.md` frontmatter 的 description。"""
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


def _apply_skill_description(langchain_tool: BaseTool, skill_id: str) -> str:
    """用官方 Skill description 覆盖 Tool 描述（模型可见）。"""
    description = skill_description(
        skill_id,
        fallback=str(getattr(langchain_tool, "description", "") or skill_id),
    )
    langchain_tool.description = description
    return description

_IWENCAI_DOMAIN = "iwencai"
_IWENCAI_DATA_TYPE = "iwencai_result"


def _finance_fact_rank(metric: str) -> int:
    """完整财年优先用累计字段；单季次之；行情垫底。"""
    if "最新价" in metric or "涨跌幅" in metric:
        return 90
    if "单季度" in metric or "单季" in metric:
        return 50
    if "累计" in metric and any(
        marker in metric for marker in ("营业收入", "营业额", "收入", "净利润", "归母", "净利")
    ):
        return 0
    if "累计" in metric:
        return 10
    return 40


def _prioritize_finance_facts(row_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同实体行内：累计财务指标优先；若已有累计营收/净利则丢掉纯行情字段。"""
    ordered = sorted(
        row_facts,
        key=lambda item: (
            _finance_fact_rank(str(item.get("metric") or "")),
            str(item.get("metric") or ""),
        ),
    )
    has_cumulative = any(
        _finance_fact_rank(str(item.get("metric") or "")) == 0 for item in ordered
    )
    if not has_cumulative:
        return ordered
    return [
        item
        for item in ordered
        if _finance_fact_rank(str(item.get("metric") or "")) < 90
    ]


def _normalize_iwencai_payload(result: dict[str, Any]) -> dict[str, Any]:
    """在工具边界补齐通用 facts；无法确认的内容保留为 unmapped_rows。"""
    payload = deepcopy(result)
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload
    rows = [row for row in list(data.get("datas") or []) if isinstance(row, dict)]
    if not rows:
        return payload

    facts: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_facts: list[dict[str, Any]] = []
        entity = ""
        for key, value in row.items():
            key_text = str(key).strip().lower()
            if key_text in {"entity", "company", "股票简称", "证券简称", "股票名称"}:
                if value not in (None, ""):
                    entity = str(value).strip()
                break

        period = ""
        for key, value in row.items():
            key_text = str(key).lower()
            if "报告期" not in key_text and "财年" not in key_text and "period" not in key_text:
                continue
            # 优先从单元格值解析日期；列名里可能带无关的字段编号后缀（如
            # "报告期截止日[20260630]": "20250930"），只用 key 兜底避免误取。
            source_text = str(value) if value not in (None, "") else str(key)
            match = re.search(r"(20\d{2})(?:[-./年]?(\d{1,2}))?(?:[-./月]?(\d{1,2}))?", source_text)
            if match:
                year, month, _day = match.groups()
                period = f"FY{year} Q4"
                if month and int(month) <= 9:
                    period = f"FY{year}"
                break

        for field_name, raw_value in row.items():
            if raw_value in (None, "") or isinstance(raw_value, bool):
                continue
            text = str(raw_value).strip().replace(",", "")
            number_match = re.match(r"[-+]?\d+(?:\.\d+)?", text)
            suffix = text[number_match.end():].strip() if number_match else ""
            if not number_match or suffix not in {"", "%", "亿", "万", "元", "亿港元", "亿元", "亿美元", "万港元", "万人民币", "美元", "港元", "人民币"}:
                continue
            if re.search(r"报告期|日期|时间|代码|股票代码|证券代码", str(field_name), re.I):
                continue
            value = float(number_match.group(0))
            unit = "%" if "%" in text else ""
            currency = ""
            for marker, code in (("美元", "USD"), ("港元", "HKD"), ("人民币", "CNY")):
                if marker in text or marker in str(field_name):
                    currency = code
                    unit = unit or marker
                    break
            # 指标名自带 [YYYYMMDD] 时，该日期直接对应本字段的取数口径，
            # 比行级 "报告期" 猜测更可靠（尤其在同一行混有不同截止日字段时）。
            fact_period = period
            bracket_match = re.search(r"\[(\d{4})(\d{2})(\d{2})\]", str(field_name))
            if bracket_match:
                bracket_year, bracket_month, _bracket_day = bracket_match.groups()
                fact_period = (
                    f"FY{bracket_year} Q4"
                    if int(bracket_month) == 12
                    else f"FY{bracket_year}"
                )
            row_facts.append({
                "entity": entity,
                "metric": str(field_name),
                "fiscal_period": fact_period,
                "value": value,
                "unit": unit,
                "currency": currency,
                "provider_field": str(field_name),
                "provider_row_index": row_index,
            })
        if row_facts:
            facts.extend(_prioritize_finance_facts(row_facts))
        else:
            unmapped_rows.append(row)

    normalized = dict(data)
    existing_facts = [item for item in list(data.get("facts") or []) if isinstance(item, dict)]
    normalized["facts"] = existing_facts + facts
    normalized["unmapped_rows"] = unmapped_rows
    normalized["coverage"] = {
        "entities": sorted({str(item.get("entity") or "") for item in facts if item.get("entity")}),
        "metrics": sorted({str(item.get("metric") or "") for item in facts if item.get("metric")}),
        "periods": sorted({str(item.get("fiscal_period") or "") for item in facts if item.get("fiscal_period")}),
    }
    payload["data"] = normalized
    return payload
_MARKET_SKILLS = frozenset(
    {
        "hithink-market-query",
        "hithink-zhishu-query",
        "hithink-industry-query",
        "hithink-finance-query",
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
        "hithink-usstock-selector",
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
    """使用自然语言查询同花顺问财数据（通用/兼容入口）。

    Args:
        query: 股票、指数、财务或选股查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数，最大值由配置控制。
    """
    result = await fetch_iwencai(query, page=page, limit=limit)
    return _normalize_iwencai_payload(result)


@tool(parse_docstring=True)
async def query_iwencai_finance(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """查询全市场个股财务指标（营收、净利、ROE、负债率、现金流等）。

    Args:
        query: 财务指标自然语言查询语句。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    result = await _run_query_skill(
        "hithink-finance-query",
        query,
        page=page,
        limit=limit,
        call_type=call_type,
    )
    return _normalize_iwencai_payload(result)


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
async def screen_iwencai_usstock(
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """通过已安装的官方问财 Skill 执行美股筛选。

    Args:
        query: 自然语言美股筛选条件。
        page: 结果页码，从 1 开始。
        limit: 每页返回条数。
        call_type: 调用类型，只能是 normal 或 retry。
    """
    return await _run_query_skill(
        "hithink-usstock-selector",
        query,
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
    """查询股票、ETF 和指数等行情（价、涨跌幅、成交、资金等）。

    不用于板块领涨或行业涨跌幅排名（请用 query_iwencai_industry）。

    Args:
        query: 行情查询语句（个股/ETF/指数；A股「今日」未收盘时请改写为明确交易日）。
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
    """查询行业估值、财务、盈利、行情和板块排名（含领涨板块/涨跌幅排名）。

    Args:
        query: 行业/板块查询语句；「今日领涨」在未收盘时请改写为系统提示中的最近已结束交易日（形如「YYYY年M月D日A股板块涨幅排名」），禁止写死某一天。
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
        description="通过同花顺问财自然语言查询股票、指数和选股数据（兼容入口）",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=query_iwencai,
    handler=query_iwencai.ainvoke,
)

# 描述一律取自 `.iwencai-skills/<skill_id>/SKILL.md` 的 description，不在此重复维护文案。
_TOOL_SKILL_BINDINGS: tuple[tuple[str, str, BaseTool, dict[str, Any]], ...] = (
    ("iwencai.finance.query", "hithink-finance-query", query_iwencai_finance, {}),
    ("iwencai.market.query", "hithink-market-query", query_iwencai_market, {}),
    ("iwencai.industry.query", "hithink-industry-query", query_iwencai_industry, {}),
    ("iwencai.index.query", "hithink-zhishu-query", query_iwencai_index, {}),
    ("iwencai.rating.query", "hithink-insresearch-query", query_iwencai_rating, {}),
    ("iwencai.announcement.search", "announcement-search", search_iwencai_announcement, {}),
    ("iwencai.report.search", "report-search", search_iwencai_report, {}),
    ("iwencai.fund.screen", "hithink-fund-selector", screen_iwencai_fund, {}),
    ("iwencai.usstock.screen", "hithink-usstock-selector", screen_iwencai_usstock, {}),
    (
        "iwencai.screen",
        "hithink-astock-selector",
        screen_iwencai,
        {"timeout_seconds": settings.IWENCAI_SKILL_RUNNER_TIMEOUT_SEC},
    ),
)

for _tool_id, _skill_id, _langchain_tool, _extra in _TOOL_SKILL_BINDINGS:
    _description = _apply_skill_description(_langchain_tool, _skill_id)
    register_tool(
        ToolSpec(
            tool_id=_tool_id,
            name=_langchain_tool.name,
            description=_description,
            risk_level="low",
            read_only=True,
            **_extra,
        ),
        langchain_tool=_langchain_tool,
        handler=_langchain_tool.ainvoke,
    )


__all__ = [
    "fetch_iwencai",
    "query_iwencai",
    "query_iwencai_finance",
    "query_iwencai_index",
    "query_iwencai_industry",
    "query_iwencai_market",
    "query_iwencai_rating",
    "screen_iwencai",
    "screen_iwencai_fund",
    "screen_iwencai_usstock",
    "search_iwencai_announcement",
    "search_iwencai_report",
    "skill_description",
]
