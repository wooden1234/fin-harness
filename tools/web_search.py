"""联网搜索工具，复用现有 web_search_agent 的提供商实现。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agents.finance_agent.web_search_agent.node import search_web as _agent_search_web
from tools.base import ToolSpec
from tools.registry import register_tool


async def fetch_web_search(query: str) -> dict[str, Any]:
    """执行联网搜索，返回结构化结果。供代码直接调用。"""
    return await _agent_search_web(query)


@tool(parse_docstring=True)
async def search_web(query: str) -> dict[str, Any]:
    """联网搜索公开网页信息，用于最新动态、公开资料补充。

    Args:
        query: 搜索关键词或完整问句
    """
    return await fetch_web_search(query)


register_tool(
    ToolSpec(
        tool_id="web.search",
        name="search_web",
        description="联网搜索公开网页信息",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=search_web,
)


__all__ = ["fetch_web_search", "search_web"]
