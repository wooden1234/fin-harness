"""高置信天气请求的确定性解析与展示。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WEATHER_MARKERS = ("天气", "气温", "下雨", "降雨")
_UNSAFE_MULTI_CITY_MARKERS = ("和", "与", "、", ",", "，")
_CITY_STOPWORDS = frozenset({"这里", "当地", "今天", "明天", "后天", "未来", "最近"})
_PREFIXES = ("请问", "请", "帮我", "帮忙", "查一下", "查询", "查查", "看看", "告诉我")
_TEMPORAL_PATTERN = re.compile(r"未来[一二三四五1-5]?天|今天|明天|后天")
_DAYS_PATTERN = re.compile(r"(?:未来)?([一二三四五1-5])天")
_CN_NUMBER = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}


@dataclass(frozen=True, slots=True)
class WeatherRequest:
    city: str
    days: int = 1


def parse_weather_request(query: str) -> WeatherRequest | None:
    """只接受单城市、显式天气意图，模糊请求继续交给通用 Agent。"""
    normalized = "".join(str(query or "").split()).strip("？?！!。.")
    marker = next((item for item in _WEATHER_MARKERS if item in normalized), None)
    if marker is None:
        return None

    days = _forecast_days(normalized)
    cleaned = _TEMPORAL_PATTERN.sub("", normalized)
    marker_index = cleaned.find(marker)
    candidate = cleaned[:marker_index]
    for prefix in _PREFIXES:
        candidate = candidate.removeprefix(prefix)
    candidate = candidate.rstrip("的会是否想要看")
    if (
        not candidate
        or candidate in _CITY_STOPWORDS
        or len(candidate) > 30
        or any(item in candidate for item in _UNSAFE_MULTI_CITY_MARKERS)
    ):
        return None
    return WeatherRequest(city=candidate, days=days)


def format_weather_result(result: dict[str, Any]) -> str:
    """把结构化天气结果投影成简短回答，不让第二次 LLM 复述工具数据。"""
    city = str(result.get("query_city") or "该城市")
    if not result.get("ok"):
        errors = {
            "not_configured": "天气服务尚未配置",
            "invalid_api_key": "天气服务认证失败",
            "city_not_found": f"没有找到“{city}”的天气位置",
            "timeout": "天气服务响应超时",
        }
        return errors.get(str(result.get("error") or ""), "天气服务暂时不可用") + "。"

    location = dict(result.get("location") or {})
    current = dict(result.get("current") or {})
    display_city = str(location.get("local_name") or location.get("name") or city)
    parts = [f"{display_city}当前{current.get('condition') or '天气情况未知'}"]
    temperature = current.get("temperature_c")
    feels_like = current.get("apparent_temperature_c")
    humidity = current.get("humidity_pct")
    if temperature is not None:
        parts.append(f"气温 {temperature}℃")
    if feels_like is not None:
        parts.append(f"体感 {feels_like}℃")
    if humidity is not None:
        parts.append(f"湿度 {humidity}%")
    answer = "，".join(parts) + "。"

    forecasts = [item for item in result.get("daily") or [] if isinstance(item, dict)]
    if forecasts:
        rows = []
        for item in forecasts:
            rows.append(
                "{date}：{condition}，{low}～{high}℃".format(
                    date=item.get("date") or "日期未知",
                    condition=item.get("condition") or "天气未知",
                    low=item.get("temp_min_c") if item.get("temp_min_c") is not None else "?",
                    high=item.get("temp_max_c") if item.get("temp_max_c") is not None else "?",
                )
            )
        answer += "\n" + "\n".join(rows)
    return answer


def _forecast_days(query: str) -> int:
    match = _DAYS_PATTERN.search(query)
    if match:
        raw = match.group(1)
        return max(1, min(5, int(raw) if raw.isdigit() else _CN_NUMBER[raw]))
    if "后天" in query:
        return 3
    if "明天" in query:
        return 2
    if "未来几天" in query:
        return 3
    return 1


__all__ = ["WeatherRequest", "format_weather_result", "parse_weather_request"]
