"""天气查询工具（OpenWeatherMap）。

``get_weather`` 经 ``@tool`` 包装，可直接交给 ``llm.bind_tools([get_weather])``，
让模型看见 schema 并选择调用；底层实现仍是 ``fetch_weather``，也可代码显式调用。

需在 ``.env`` 中配置 ``OPENWEATHER_API_KEY``（及可选的 URL / 超时 / 天数上限）。
中文城市名会先经 Geocoding API 解析为经纬度，再查天气。
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from time import perf_counter
from typing import Any

import httpx
from langchain_core.tools import tool

from app.core.config import settings
from app.core.logger import get_logger
from tools.base import ToolSpec
from tools.registry import register_tool

logger = get_logger(service="weather_tool")


def _api_key() -> str:
    return (settings.OPENWEATHER_API_KEY or "").strip()


def _auth_params() -> dict[str, str]:
    return {
        "appid": _api_key(),
        "units": "metric",
        "lang": "zh_cn",
    }


def _pick_weather(item: dict[str, Any]) -> dict[str, Any]:
    weather_list = list(item.get("weather") or [])
    weather = weather_list[0] if weather_list else {}
    return {
        "weather_id": weather.get("id"),
        "condition": str(weather.get("description") or weather.get("main") or "未知"),
        "main": str(weather.get("main") or ""),
        "icon": str(weather.get("icon") or ""),
    }


async def _geocode_city(
    client: httpx.AsyncClient,
    city: str,
) -> dict[str, Any] | None:
    response = await client.get(
        settings.OPENWEATHER_GEOCODE_URL,
        params={"q": city, "limit": 1, "appid": _api_key()},
    )
    if response.status_code == 401:
        raise PermissionError("invalid_api_key")
    response.raise_for_status()
    results = list(response.json() or [])
    if not results:
        return None
    hit = results[0]
    local_names = dict(hit.get("local_names") or {})
    return {
        "name": str(hit.get("name") or city),
        "local_name": str(local_names.get("zh") or hit.get("name") or city),
        "country": str(hit.get("country") or ""),
        "state": str(hit.get("state") or ""),
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
    }


def _aggregate_daily(forecast_list: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    """将 3 小时预报按日期聚合成逐日摘要。"""
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in forecast_list:
        dt_txt = str(item.get("dt_txt") or "")
        if not dt_txt or " " not in dt_txt:
            continue
        date = dt_txt.split(" ", 1)[0]
        by_date[date].append(item)

    daily: list[dict[str, Any]] = []
    for date in sorted(by_date.keys())[:days]:
        slots = by_date[date]
        temps = [
            float(slot["main"]["temp"])
            for slot in slots
            if isinstance(slot.get("main"), dict) and slot["main"].get("temp") is not None
        ]
        rains = [
            float((slot.get("rain") or {}).get("3h") or 0)
            for slot in slots
            if isinstance(slot, dict)
        ]
        noonish = min(
            slots,
            key=lambda s: abs(int(str(s.get("dt_txt") or "00:00").split(" ")[-1][:2] or 0) - 12),
        )
        weather = _pick_weather(noonish)
        daily.append(
            {
                "date": date,
                "temp_max_c": round(max(temps), 1) if temps else None,
                "temp_min_c": round(min(temps), 1) if temps else None,
                "precipitation_mm": round(sum(rains), 1) if rains else 0.0,
                "condition": weather["condition"],
                "weather_id": weather["weather_id"],
                "main": weather["main"],
            }
        )
    return daily


async def fetch_weather(
    city: str,
    days: int = 1,
) -> dict[str, Any]:
    """按城市查询当前天气；``days>1`` 时附带逐日预报（上限见 OPENWEATHER_MAX_DAYS）。

    返回结构化 dict，不生成面向用户的完整自然语言答案。
    供代码直接调用；模型侧请使用 ``get_weather``（@tool）。
    """
    city = (city or "").strip()
    if not city:
        return {"ok": False, "error": "empty_city", "city": ""}

    if not _api_key():
        return {
            "ok": False,
            "error": "not_configured",
            "city": city,
            "detail": "未配置 OPENWEATHER_API_KEY，请在 .env 中设置",
        }

    max_days = max(1, int(settings.OPENWEATHER_MAX_DAYS or 5))
    forecast_days = max(1, min(int(days or 1), max_days))

    try:
        async with httpx.AsyncClient(timeout=settings.OPENWEATHER_TIMEOUT_SEC) as client:
            try:
                location = await _geocode_city(client, city)
            except PermissionError:
                return {
                    "ok": False,
                    "error": "invalid_api_key",
                    "city": city,
                    "detail": "OpenWeather API Key 无效或未激活",
                }
            if location is None:
                return {"ok": False, "error": "city_not_found", "city": city}

            coords = {
                "lat": location["latitude"],
                "lon": location["longitude"],
                **_auth_params(),
            }

            current_resp = await client.get(
                settings.OPENWEATHER_CURRENT_URL, params=coords
            )
            if current_resp.status_code == 401:
                return {
                    "ok": False,
                    "error": "invalid_api_key",
                    "city": city,
                    "detail": "OpenWeather API Key 无效或未激活",
                }
            current_resp.raise_for_status()
            current_payload = current_resp.json() or {}

            daily: list[dict[str, Any]] = []
            if forecast_days > 1:
                forecast_resp = await client.get(
                    settings.OPENWEATHER_FORECAST_URL, params=coords
                )
                forecast_resp.raise_for_status()
                forecast_payload = forecast_resp.json() or {}
                daily = _aggregate_daily(
                    list(forecast_payload.get("list") or []),
                    days=forecast_days,
                )
                location["timezone_offset_sec"] = forecast_payload.get("city", {}).get(
                    "timezone"
                )
            else:
                location["timezone_offset_sec"] = current_payload.get("timezone")
    except httpx.TimeoutException:
        return {"ok": False, "error": "timeout", "city": city}
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "http_error",
            "city": city,
            "detail": str(exc),
        }

    main = dict(current_payload.get("main") or {})
    wind = dict(current_payload.get("wind") or {})
    weather = _pick_weather(current_payload)

    current = {
        "time": current_payload.get("dt"),
        "temperature_c": main.get("temp"),
        "apparent_temperature_c": main.get("feels_like"),
        "humidity_pct": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed_ms": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "condition": weather["condition"],
        "weather_id": weather["weather_id"],
        "main": weather["main"],
        "icon": weather["icon"],
    }

    return {
        "ok": True,
        "provider": "openweathermap",
        "query_city": city,
        "location": location,
        "current": current,
        "daily": daily,
        "units": {
            "temperature": "celsius",
            "wind_speed": "m/s",
            "precipitation": "mm",
            "pressure": "hPa",
        },
    }


@tool(parse_docstring=True)
async def get_weather(city: str, days: int = 1) -> dict[str, Any]:
    """查询指定城市的当前天气与近几日预报。

    Args:
        city: 城市名称，例如「上海」「Beijing」「深圳」
        days: 预报天数；1 表示仅当前天气，上限见 OPENWEATHER_MAX_DAYS
    """
    started_at = perf_counter()
    log_city = " ".join((city or "").split())[:80]
    logger.info("get_weather started city={} days={}", log_city, days)
    try:
        result = await fetch_weather(city, days=days)
    except asyncio.CancelledError:
        logger.warning(
            "get_weather cancelled city={} days={} elapsed_ms={:.1f}",
            log_city,
            days,
            (perf_counter() - started_at) * 1000,
        )
        raise
    except Exception as exc:
        logger.warning(
            "get_weather failed city={} days={} elapsed_ms={:.1f} error_type={}",
            log_city,
            days,
            (perf_counter() - started_at) * 1000,
            type(exc).__name__,
        )
        raise

    logger.info(
        "get_weather finished city={} days={} elapsed_ms={:.1f} ok={} error={}",
        log_city,
        days,
        (perf_counter() - started_at) * 1000,
        bool(result.get("ok")),
        str(result.get("error") or ""),
    )
    return result


register_tool(
    ToolSpec(
        tool_id="weather.get",
        name="get_weather",
        description="查询指定城市的当前天气与近几日预报（OpenWeather）",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=get_weather,
    handler=get_weather.ainvoke,
)


__all__ = ["fetch_weather", "get_weather"]
