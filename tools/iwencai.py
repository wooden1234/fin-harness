"""同花顺问财 SkillHub OpenAPI 工具。

该模块只负责把问财查询封装为受治理的只读 Tool，不在这里解析投资建议，
也不把 API Key 写入返回值、日志或 LangGraph 状态。
"""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_core.tools import tool

from app.core.config import settings
from skills.runners.iwencai import run_installed_skill
from tools.base import ToolSpec
from tools.registry import register_tool

_SKILL_ID = "hithink-astock-selector"
_SKILL_VERSION = "1.0.0"


def _base_url() -> str:
    return settings.IWENCAI_BASE_URL.rstrip("/") + "/"


def _api_key() -> str:
    return settings.IWENCAI_API_KEY.strip()


def _validate_limit(limit: int) -> int:
    if limit < 1 or limit > settings.IWENCAI_MAX_LIMIT:
        raise ValueError(f"limit_must_be_between_1_and_{settings.IWENCAI_MAX_LIMIT}")
    return limit


async def fetch_iwencai(
    query: str,
    *,
    page: int = 1,
    limit: int = 10,
) -> dict[str, Any]:
    """调用问财结构化查询接口，返回原始数据和最小元数据。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query_must_not_be_empty")
    if page < 1:
        raise ValueError("page_must_be_positive")
    normalized_limit = _validate_limit(limit)

    api_key = _api_key()
    if not api_key:
        return {
            "ok": False,
            "error": "iwencai_not_configured",
            "message": "请配置 IWENCAI_API_KEY 后再调用问财工具。",
        }

    payload = {
        "query": normalized_query,
        "page": str(page),
        "limit": str(normalized_limit),
        "is_cache": "1",
        "expand_index": "true",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Claw-Call-Type": "normal",
        "X-Claw-Skill-Id": _SKILL_ID,
        "X-Claw-Skill-Version": _SKILL_VERSION,
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
        "query": normalized_query,
        "page": page,
        "limit": normalized_limit,
        "data": body.get("data", body),
    }


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
    return await run_installed_skill(
        "hithink-astock-selector",
        query=query,
        page=page,
        limit=limit,
        call_type=call_type,
    )


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


__all__ = ["fetch_iwencai", "query_iwencai", "screen_iwencai"]
