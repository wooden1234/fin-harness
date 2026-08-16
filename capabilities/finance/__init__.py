"""本地财务事实。"""

from __future__ import annotations

from typing import Any

from capabilities.evidence import stamp_evidence


async def lookup_fact(arguments: dict[str, Any]) -> dict[str, Any]:
    from tools.finance import fetch_financial_fact

    result = await fetch_financial_fact(
        str(arguments.get("question") or ""),
        list(arguments.get("companies") or []),
        list(arguments.get("metrics") or []),
        years=arguments.get("years"),
        operation=arguments.get("operation") or "latest",
        top_k=int(arguments.get("top_k") or 5),
    )
    payload = result if isinstance(result, dict) else {"ok": True, "data": result}
    payload.setdefault("ok", True)
    return stamp_evidence("knowledge.fact.lookup", payload)
