"""天气查询工具（OpenWeatherMap）。

``get_weather`` 经 ``@tool`` 包装，可直接交给 ``llm.bind_tools([get_weather])``，
让模型看见 schema 并选择调用；底层实现仍是 ``fetch_weather``，也可代码显式调用。

需在 ``.env`` 中配置 ``OPENWEATHER_API_KEY``（及可选的 URL / 超时）。
中文城市名会先经 Geocoding API 解析为经纬度，再查天气。
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

import httpx
from langchain_core.tools import tool

from app.core.cache import (
    cache_get,
    cache_set,
    make_record,
    normalize_cache_text,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis_keys import redis_keys
from tools.core.base import ToolSpec
from tools.core.registry import register_tool

logger = get_logger(service="weather_tool")

_WEATHER_CACHE_DOMAIN = "weather"
_WEATHER_CACHE_DATA_TYPE = "weather_current_v1"


def _weather_cache_key(city: str):
    normalized_city = normalize_cache_text(city).casefold()
    return redis_keys.build(
        _WEATHER_CACHE_DOMAIN,
        "current",
        redis_keys.digest(normalized_city),
    )

# 行政区划后缀（语言学结构，非城市名单）：去掉后再做命中校验。
_ADMIN_SUFFIXES: tuple[str, ...] = (
    "特别行政区",
    "自治州",
    "自治区",
    "地区",
    "盟",
    "省",
    "市",
    "州",
    "区",
    "县",
)


def _admin_suffix(place: str) -> str | None:
    """识别行政区划后缀；按长后缀优先匹配。"""
    normalized = "".join((place or "").split())
    for suffix in _ADMIN_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            return suffix
    return None


def _normalize_place_query(city: str) -> str:
    normalized = "".join((city or "").split())
    suffix = _admin_suffix(normalized)
    if suffix:
        return normalized[: -len(suffix)]
    return normalized


def _admin_level_bonus(query: str, local_zh: str) -> float:
    """按行政区划层级给分，避免「西安市」与「西安区」同分误判歧义。

    - 用户未写区/县时：优先「…市」，弱化同名「…区/县」。
    - 用户显式写了区/县：优先同级后缀命中。
    """
    hit_suffix = _admin_suffix(local_zh)
    if not hit_suffix:
        return 0.0
    query_suffix = _admin_suffix(query)
    if query_suffix in {"区", "县"}:
        if hit_suffix == query_suffix:
            return 1.25
        if hit_suffix == "市":
            return 0.2
        return 0.3
    if hit_suffix == "市":
        return 1.0
    if hit_suffix in {"区", "县"}:
        return 0.0
    # 州/盟/地区等其它正式后缀，略优于裸地名村镇。
    return 0.5


def _cjk_chars(text: str) -> str:
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")


def _latin_fold(text: str) -> str:
    cleaned = []
    for ch in (text or "").casefold():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        elif ch in {"'", "’", "-", "_"}:
            continue
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _geocode_candidates(city: str) -> list[str]:
    """仅做结构归一：原文 + 去行政区划后缀；不做城市/省份别名表。"""
    raw = (city or "").strip()
    normalized = _normalize_place_query(raw)
    ordered: list[str] = []
    for item in (raw, normalized):
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def _hit_text_blob(hit: dict[str, Any]) -> str:
    local_names = hit.get("local_names") or {}
    parts: list[str] = [
        str(hit.get("name") or ""),
        str(hit.get("state") or ""),
        str(hit.get("country") or ""),
    ]
    if isinstance(local_names, dict):
        parts.extend(str(v) for v in local_names.values() if v)
    return " ".join(parts)


def _score_geocode_hit(query: str, hit: dict[str, Any]) -> float | None:
    """通用命中校验：查询词字符须落在命中文本中；失败返回 None。

    - 中文：归一化后的每个汉字都必须出现在命中的中文文本里（避免陕西→山西）。
    - 拉丁：查询词需与 name/state 形成包含或高重叠关系。
    """
    if not isinstance(hit, dict):
        return None
    try:
        float(hit["lat"])
        float(hit["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    q_norm = _normalize_place_query(query)
    if not q_norm:
        return None

    blob = _hit_text_blob(hit)
    q_cjk = _cjk_chars(q_norm)
    blob_cjk = _cjk_chars(blob)
    local_zh = ""
    local_names = hit.get("local_names") or {}
    if isinstance(local_names, dict):
        local_zh = str(local_names.get("zh") or "")

    if q_cjk:
        if not blob_cjk:
            return None
        if any(ch not in blob_cjk for ch in q_cjk):
            return None
        score = 1.0
        normalized_local_zh = _cjk_chars(_normalize_place_query(local_zh))
        if local_zh and q_cjk == normalized_local_zh:
            score += 3.0
            # 「上海市」优于同名村镇；「西安市」优于同名「西安区」。
            score += _admin_level_bonus(query, local_zh)
        elif q_cjk in blob_cjk:
            score += 2.0
        else:
            # 部分覆盖：按字符命中密度给分（已保证全覆盖）
            score += len(q_cjk) / max(len(blob_cjk), 1)
        name = str(hit.get("name") or "")
        if q_cjk == _cjk_chars(name):
            score += 1.0
        return score

    q_latin = _latin_fold(q_norm)
    if len(q_latin) < 2:
        return None
    name_latin = _latin_fold(str(hit.get("name") or ""))
    state_latin = _latin_fold(str(hit.get("state") or ""))
    blob_latin = _latin_fold(blob)
    if not name_latin and not state_latin:
        return None
    if q_latin == name_latin:
        return 4.0
    if name_latin.startswith(q_latin) or q_latin.startswith(name_latin):
        return 3.0
    if q_latin in name_latin or q_latin in state_latin or q_latin in blob_latin:
        return 2.0
    # token overlap
    q_tokens = set(q_latin.split())
    hit_tokens = set(blob_latin.split())
    if q_tokens and q_tokens <= hit_tokens:
        return 1.5
    return None


def _location_from_hit(hit: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    local_names = dict(hit.get("local_names") or {})
    return {
        "name": str(hit.get("name") or fallback_name),
        "local_name": str(local_names.get("zh") or hit.get("name") or fallback_name),
        "country": str(hit.get("country") or ""),
        "state": str(hit.get("state") or ""),
        "latitude": float(hit["lat"]),
        "longitude": float(hit["lon"]),
    }


def _candidate_preview(hit: dict[str, Any]) -> dict[str, str]:
    local_names = dict(hit.get("local_names") or {})
    return {
        "name": str(hit.get("name") or ""),
        "local_name": str(local_names.get("zh") or ""),
        "state": str(hit.get("state") or ""),
        "country": str(hit.get("country") or ""),
    }


def _pick_geocode_hit(
    query: str,
    results: list[Any],
) -> tuple[dict[str, Any] | None, list[dict[str, str]], str | None]:
    """从 geocode 结果中选命中。

    返回 (location|None, candidate_previews, reject_reason|None)。
    reject_reason:
      - unresolved: 有结果但无一通过校验（常见于省份/错配）
      - ambiguous: 多个高分命中难分
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for hit in results:
        if not isinstance(hit, dict):
            continue
        score = _score_geocode_hit(query, hit)
        if score is None:
            continue
        scored.append((score, hit))

    if not scored:
        return None, [], "unresolved" if results else None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_hit = scored[0]
    # 同分/近分且地名不同 → 需用户补具体城市
    rivals = [
        hit
        for score, hit in scored
        if score >= best_score - 0.35
    ]
    distinct = []
    seen: set[tuple[str, str, str, float, float]] = set()
    for hit in rivals:
        key = (
            str(hit.get("name") or ""),
            str(hit.get("state") or ""),
            str(hit.get("country") or ""),
            round(float(hit.get("lat") or 0), 2),
            round(float(hit.get("lon") or 0), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        distinct.append(hit)

    if len(distinct) > 1:
        # 不同 state 或 country 才算真歧义。
        states = {str(hit.get("state") or "") for hit in distinct}
        countries = {str(hit.get("country") or "") for hit in distinct}
        if len(states) > 1 or len(countries) > 1:
            return (
                None,
                [_candidate_preview(hit) for hit in distinct[:5]],
                "ambiguous",
            )

    return _location_from_hit(best_hit, query), [], None


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
    """解析城市坐标。校验失败或歧义时返回带 error 语义的占位 dict。

    成功：普通 location dict（含 lat/lon）。
    需补城市：{"error": "need_city", "reason": ..., "candidates": [...]}。
    未找到：返回 None（由上层映射 city_not_found）。
    """
    last_error: Exception | None = None
    saw_results = False
    for candidate in _geocode_candidates(city):
        try:
            response = await _get_with_retry(
                client,
                settings.OPENWEATHER_GEOCODE_URL,
                params={"q": candidate, "limit": 5, "appid": _api_key()},
            )
            if response.status_code == 401:
                raise PermissionError("invalid_api_key")
            response.raise_for_status()
            results = list(response.json() or [])
            if results:
                saw_results = True
            location, previews, reason = _pick_geocode_hit(city, results)
            if location is not None:
                return location
            if reason == "ambiguous":
                return {
                    "error": "need_city",
                    "reason": "ambiguous",
                    "candidates": previews,
                }
            if reason == "unresolved":
                # 保留原始命中供排查；不把错配地点塞给模型当可选项。
                candidates = [
                    _candidate_preview(hit)
                    for hit in results
                    if isinstance(hit, dict)
                ][:5]
                logger.warning(
                    "geocode unresolved city={} candidate={} raw={}",
                    city,
                    candidate,
                    candidates,
                )
                continue
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "geocode failed city={} candidate={} error={}",
                city,
                candidate,
                type(exc).__name__,
            )
            continue

    if saw_results:
        return {
            "error": "need_city",
            "reason": "unresolved",
            "candidates": [],
            "detail": "地点与地理库命中不一致或过于宽泛，请补充具体城市",
        }
    if last_error is not None:
        logger.warning(
            "geocode exhausted city={} last_error={}",
            city,
            type(last_error).__name__,
        )
    return None


def _current_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """解析 OpenWeather 2.5 当前天气响应。"""
    main = payload.get("main")
    if not isinstance(main, dict) or main.get("temp") is None:
        return None
    wind = payload.get("wind") if isinstance(payload.get("wind"), dict) else {}
    system = payload.get("sys") if isinstance(payload.get("sys"), dict) else {}
    weather = _pick_weather(payload)
    return {
        "time": payload.get("dt"),
        "sunrise": system.get("sunrise"),
        "sunset": system.get("sunset"),
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


def _weather_api_error(
    status_code: int,
    payload: Any,
    city: str,
) -> dict[str, Any] | None:
    """识别 OpenWeather 2.5 的鉴权及其他业务错误。"""
    body = payload if isinstance(payload, dict) else {}
    code = str(body.get("cod") or body.get("code") or "")

    if status_code == 401 or code == "401":
        return {
            "ok": False,
            "error": "invalid_api_key",
            "city": city,
            "detail": "OpenWeather API Key 无效或未激活",
        }

    if status_code >= 400 or (code and code not in {"200", "0"}):
        return {
            "ok": False,
            "error": "weather_api_error",
            "city": city,
            "detail": "OpenWeather 天气服务返回错误",
        }
    return None


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any],
    attempts: int = 2,
) -> httpx.Response:
    """OpenWeather 跨境链路不稳：超时后立刻重试一次。"""
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            last_exc = exc
            logger.warning(
                "openweather timeout url={} attempt={}/{}",
                url,
                attempt + 1,
                attempts,
            )
            if attempt + 1 >= attempts:
                raise
    assert last_exc is not None
    raise last_exc


async def fetch_weather(
    city: str,
) -> dict[str, Any]:
    """按城市查询当前天气；当前仅支持单日查询。

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

    cache_enabled = bool(settings.WEATHER_CACHE_ENABLED)
    cache_key = _weather_cache_key(city)
    cached = await cache_get(
        cache_key,
        domain=_WEATHER_CACHE_DOMAIN,
        data_type=_WEATHER_CACHE_DATA_TYPE,
        enabled=cache_enabled,
    )
    if cached is not None and cached.kind == "record" and isinstance(cached.payload, dict):
        result = dict(cached.payload)
        result["cache_status"] = "hit"
        return result

    try:
        request_timeout = httpx.Timeout(
            max(12.0, float(settings.OPENWEATHER_TIMEOUT_SEC) + 2.0),
            connect=6.0,
        )
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            try:
                location = await _geocode_city(client, city)
            except PermissionError:
                return {
                    "ok": False,
                    "error": "invalid_api_key",
                    "city": city,
                    "detail": "OpenWeather API Key 无效或未激活",
                }
            if isinstance(location, dict) and location.get("error") == "need_city":
                return {
                    "ok": False,
                    "error": "need_city",
                    "city": city,
                    "reason": str(location.get("reason") or "unresolved"),
                    "candidates": list(location.get("candidates") or []),
                    "detail": str(
                        location.get("detail")
                        or "地点不够具体或存在歧义，请补充具体城市名"
                    ),
                }
            if location is None:
                return {"ok": False, "error": "city_not_found", "city": city}

            coords = {
                "lat": location["latitude"],
                "lon": location["longitude"],
                **_auth_params(),
            }
            weather_resp = await _get_with_retry(
                client,
                settings.OPENWEATHER_CURRENT_URL,
                params=coords,
            )
            try:
                weather_payload = weather_resp.json() or {}
            except ValueError:
                weather_payload = {}
            api_error = _weather_api_error(
                weather_resp.status_code,
                weather_payload,
                city,
            )
            if api_error is not None:
                return api_error
            weather_resp.raise_for_status()

            location["timezone_offset_sec"] = weather_payload.get("timezone")
    except httpx.TimeoutException:
        return {"ok": False, "error": "timeout", "city": city}
    except httpx.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "ok": False,
            "error": "http_error",
            "city": city,
            "detail": (
                f"OpenWeather HTTP 请求失败（状态码 {status_code}）"
                if status_code is not None
                else "OpenWeather HTTP 请求失败"
            ),
        }

    current = _current_from_payload(weather_payload)
    if current is None:
        return {
            "ok": False,
            "error": "invalid_response",
            "city": city,
            "detail": "OpenWeather 2.5 返回中缺少当前天气记录",
        }

    result = {
        "ok": True,
        "provider": "openweathermap",
        "query_city": city,
        "location": location,
        "current": current,
        "units": {
            "temperature": "celsius",
            "wind_speed": "m/s",
            "precipitation": "mm",
            "pressure": "hPa",
        },
    }
    await cache_set(
        cache_key,
        make_record(
            data_type=_WEATHER_CACHE_DATA_TYPE,
            payload=result,
        ),
        domain=_WEATHER_CACHE_DOMAIN,
        ttl_seconds=int(settings.WEATHER_CACHE_TTL_SEC),
        enabled=cache_enabled,
        max_bytes=int(settings.WEATHER_CACHE_MAX_BYTES),
    )
    result["cache_status"] = "miss"
    return result


@tool(parse_docstring=True)
async def get_weather(city: str) -> dict[str, Any]:
    """查询指定城市的当前天气；当前仅支持单日查询。

    Args:
        city: 城市名称，例如「上海」「Beijing」「深圳」
    """
    started_at = perf_counter()
    log_city = " ".join((city or "").split())[:80]
    logger.info("get_weather started city={}", log_city)
    try:
        result = await fetch_weather(city)
    except asyncio.CancelledError:
        logger.warning(
            "get_weather cancelled city={} elapsed_ms={:.1f}",
            log_city,
            (perf_counter() - started_at) * 1000,
        )
        raise
    except Exception as exc:
        logger.warning(
            "get_weather failed city={} elapsed_ms={:.1f} error_type={}",
            log_city,
            (perf_counter() - started_at) * 1000,
            type(exc).__name__,
        )
        raise

    logger.info(
        "get_weather finished city={} elapsed_ms={:.1f} ok={} error={}",
        log_city,
        (perf_counter() - started_at) * 1000,
        bool(result.get("ok")),
        str(result.get("error") or ""),
    )
    return result


register_tool(
    ToolSpec(
        tool_id="weather.get",
        name="get_weather",
        description="查询指定城市的当前天气（OpenWeather 2.5）",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=get_weather,
    handler=get_weather.ainvoke,
)


__all__ = ["fetch_weather", "get_weather"]
