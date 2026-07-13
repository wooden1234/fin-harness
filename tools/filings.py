"""公告和监管文件工具占位。"""

from __future__ import annotations


async def search_filings(query: str) -> dict:
    return {"query": query, "status": "not_configured", "results": []}
