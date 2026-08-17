"""把工具调用/结果收成前端可渲染的步骤摘要，不透传原始 payload。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def sanitize_step_detail(raw: object) -> dict[str, Any] | None:
    """只保留前端可渲染的展示字段。"""
    if not isinstance(raw, Mapping):
        return None
    detail: dict[str, Any] = {}
    title = str(raw.get("title") or "").strip()[:80]
    query = str(raw.get("query") or "").strip()[:200]
    display_text = str(raw.get("display_text") or "").strip()[:800]
    error = str(raw.get("error") or "").strip()[:120]
    if title:
        detail["title"] = title
    if query:
        detail["query"] = query
    if display_text and not display_text.startswith(("{", "[")):
        detail["display_text"] = display_text
    if error:
        detail["error"] = error
    columns = raw.get("columns")
    rows = raw.get("rows")
    if isinstance(columns, list) and isinstance(rows, list) and columns and rows:
        clean_columns = [str(item).strip()[:40] for item in columns[:6] if str(item).strip()]
        clean_rows: list[list[str]] = []
        for row in rows[:8]:
            if not isinstance(row, (list, tuple)):
                continue
            cells = [str(cell).strip()[:80] for cell in list(row)[: len(clean_columns)]]
            if len(cells) == len(clean_columns) and any(cells):
                clean_rows.append(cells)
        if clean_columns and clean_rows:
            detail["columns"] = clean_columns
            detail["rows"] = clean_rows
    return detail or None


def _parse_json(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _query_from_mapping(payload: Mapping[str, Any]) -> str:
    for key in ("query", "city", "q", "query_city"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def detail_from_tool_call(arguments: Any) -> dict[str, Any] | None:
    parsed = _parse_json(arguments)
    if not isinstance(parsed, Mapping):
        return None
    query = _query_from_mapping(parsed)
    return sanitize_step_detail({"query": query} if query else None)


def _fmt_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text
    return str(value).strip()


def _facts_table(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    if not isinstance(data, Mapping):
        return None
    facts = [item for item in list(data.get("facts") or []) if isinstance(item, Mapping)]
    if facts:
        rows: list[list[str]] = []
        for item in facts[:8]:
            entity = str(item.get("entity") or "").strip()
            metric = str(item.get("metric") or "").strip()
            unit = str(item.get("unit") or "").strip()
            number = _fmt_number(item.get("value"))
            value = f"{number}{unit}" if number else ""
            if not (entity or metric or value):
                continue
            rows.append([entity or "—", metric or "—", value or "—"])
        if rows:
            return {"columns": ["主体", "指标", "数值"], "rows": rows}

    datas = [item for item in list(data.get("datas") or []) if isinstance(item, Mapping)]
    if not datas:
        return None
    keys = [str(key).strip() for key in list(datas[0].keys())[:6] if str(key).strip()]
    if not keys:
        return None
    rows = []
    for item in datas[:8]:
        cells = [str(item.get(key) if item.get(key) is not None else "").strip()[:80] for key in keys]
        if any(cells):
            rows.append(cells)
    if not rows:
        return None
    return {"columns": keys, "rows": rows}


def _weather_text(payload: Mapping[str, Any]) -> str:
    location = payload.get("location") if isinstance(payload.get("location"), Mapping) else {}
    current = payload.get("current") if isinstance(payload.get("current"), Mapping) else {}
    city = str(
        (location or {}).get("name")
        or payload.get("query_city")
        or payload.get("city")
        or ""
    ).strip()
    condition = str((current or {}).get("condition") or "").strip()
    temp = (current or {}).get("temperature_c")
    parts = [part for part in (city, condition) if part]
    if temp not in (None, ""):
        parts.append(f"{_fmt_number(temp)}°C")
    return " ".join(parts)


def _web_text(payload: Mapping[str, Any]) -> str:
    answer = str(payload.get("answer") or "").strip()
    if answer:
        return answer[:800]
    results = [item for item in list(payload.get("results") or []) if isinstance(item, Mapping)]
    titles = [str(item.get("title") or "").strip() for item in results[:3]]
    titles = [title for title in titles if title]
    return "；".join(titles)


def _knowledge_text(payload: Mapping[str, Any]) -> str:
    for key in ("answer", "content", "text", "snippet"):
        text = str(payload.get(key) or "").strip()
        if text and not text.startswith(("{", "[")):
            return text[:800]
    hits = payload.get("hits") or payload.get("results") or payload.get("items")
    if isinstance(hits, list):
        snippets: list[str] = []
        for item in hits[:3]:
            if isinstance(item, Mapping):
                snippet = str(item.get("text") or item.get("content") or item.get("title") or "").strip()
            else:
                snippet = str(item).strip()
            if snippet:
                snippets.append(snippet[:160])
        if snippets:
            return "；".join(snippets)
    return ""


def detail_from_tool_result(
    *,
    name: str,
    content: Any,
    ok: bool,
) -> dict[str, Any] | None:
    parsed = _parse_json(content)
    payload = parsed if isinstance(parsed, Mapping) else {}
    query = _query_from_mapping(payload)
    raw: dict[str, Any] = {}
    if query:
        raw["query"] = query
    if not ok:
        error = str(payload.get("detail") or payload.get("error") or payload.get("message") or "").strip()
        if error:
            raw["error"] = error
        return sanitize_step_detail(raw)

    lowered = (name or "").lower()
    if "iwencai" in lowered or "finance" in lowered:
        table = _facts_table(payload)
        if table:
            raw.update(table)
    elif "weather" in lowered:
        text = _weather_text(payload)
        if text:
            raw["display_text"] = text
    elif lowered.startswith("web") or "search_web" in lowered:
        text = _web_text(payload)
        if text:
            raw["display_text"] = text
    else:
        text = _knowledge_text(payload)
        if text:
            raw["display_text"] = text
        else:
            table = _facts_table(payload)
            if table:
                raw.update(table)
    return sanitize_step_detail(raw)
