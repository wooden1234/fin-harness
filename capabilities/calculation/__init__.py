"""计算。"""

from __future__ import annotations

from typing import Any

from capabilities.evidence import stamp_evidence


async def run_calculation(arguments: dict[str, Any]) -> dict[str, Any]:
    from tools.calculation import run_calculation as _run

    result = await _run(arguments)
    payload = result if isinstance(result, dict) else {"ok": True, "data": result}
    payload.setdefault("ok", True)
    return stamp_evidence("calculation.run", payload)
