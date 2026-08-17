"""问财。"""

from __future__ import annotations

from typing import Any

from capabilities.evidence import stamp_evidence


async def run_iwencai(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from tools.core.registry import get_registered_tool

    entry = get_registered_tool(tool_id)
    result = await entry.handler(arguments)
    payload = result if isinstance(result, dict) else {"ok": True, "data": result}
    payload.setdefault("ok", True)
    return stamp_evidence(tool_id, payload)
