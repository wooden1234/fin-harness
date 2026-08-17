"""天气。"""

from __future__ import annotations

from typing import Any

from capabilities.evidence import stamp_evidence


async def get_current_weather(city: str) -> dict[str, Any]:
    from tools.weather import fetch_weather

    result = await fetch_weather(city)
    payload = result if isinstance(result, dict) else {"ok": True, "data": result}
    payload.setdefault("ok", True)
    return stamp_evidence("weather.get", payload)
