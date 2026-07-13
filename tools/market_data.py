"""行情数据工具占位。"""

from __future__ import annotations


async def get_quote(symbol: str) -> dict:
    return {"symbol": symbol, "status": "not_configured"}
