"""联网搜索工具，复用现有 web_search_agent 的提供商实现。"""

from __future__ import annotations

from agents.finance_agent.web_search_agent.node import search_web as _agent_search_web


async def search_web(query: str):
    return await _agent_search_web(query)
