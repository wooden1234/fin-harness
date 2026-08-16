"""解析模型粘连的 tool arguments，避免 Pydantic 收到 ``_raw``。"""

from __future__ import annotations

import json
from typing import Any, Mapping


def parse_json_objects(raw: Any) -> list[dict[str, Any]]:
    """解码一个或多个首尾相接的 JSON 对象；兼容 ``_raw`` 包裹。"""
    if isinstance(raw, dict):
        if set(raw.keys()) == {"_raw"}:
            return parse_json_objects(raw.get("_raw"))
        nested = raw.get("_raw")
        if isinstance(nested, str) and "query" not in raw:
            parsed = parse_json_objects(nested)
            if parsed:
                return parsed
        return [dict(raw)]
    if isinstance(raw, (list, tuple)):
        objects: list[dict[str, Any]] = []
        for item in raw:
            objects.extend(parse_json_objects(item))
        return objects
    text = str(raw or "").strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    index = 0
    objects = []
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            parsed, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        objects.extend(parse_json_objects(parsed))
        index = end
    return objects


def schema_property_names(openai_schema: Mapping[str, Any] | None) -> set[str]:
    if not openai_schema:
        return set()
    function = openai_schema.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else openai_schema.get("parameters")
    if not isinstance(parameters, Mapping):
        return set()
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return set()
    return {str(name) for name in properties}


def filter_tool_arguments(
    payload: Mapping[str, Any],
    *,
    openai_schema: Mapping[str, Any] | None,
) -> dict[str, Any]:
    allowed = schema_property_names(openai_schema)
    cleaned = {key: value for key, value in payload.items() if key != "_raw"}
    if not allowed:
        return cleaned
    return {key: value for key, value in cleaned.items() if key in allowed}


def coerce_tool_arguments(
    raw: Any,
    *,
    openai_schema: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """得到可直接交给 StructuredTool 的参数列表；粘连 JSON 拆成多次调用。"""
    payloads = []
    for item in parse_json_objects(raw):
        payloads.append(filter_tool_arguments(item, openai_schema=openai_schema))
    return payloads


def merge_tool_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    items = [dict(item) for item in results]
    if not items:
        return {"ok": False, "error": "malformed_arguments"}
    if len(items) == 1:
        return items[0]
    rows: list[Any] = []
    for item in items:
        data = item.get("data")
        if isinstance(data, dict):
            rows.extend([row for row in list(data.get("datas") or []) if row])
    merged: dict[str, Any] = {
        "ok": all(item.get("ok", True) for item in items),
        "batched": True,
        "results": items,
    }
    if rows:
        merged["data"] = {"datas": rows}
    if not merged["ok"]:
        merged["error"] = "batched_tool_partial_failure"
    return merged
