"""把工具结果投影成放得进上下文的完整记录，原文仍留在日志里。"""

from __future__ import annotations

import json
from typing import Any

_SUMMARY_CHARS = 120
_READ_SUMMARY_CHARS = 480
_TITLE_CHARS = 200


def _is_record(item: Any) -> bool:
    return isinstance(item, dict) and any(
        isinstance(item.get(key), str) and item.get(key)
        for key in ("title", "summary")
    )


def _record_lists(node: Any) -> list[tuple[dict[str, Any], str]]:
    found: list[tuple[dict[str, Any], str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                isinstance(value, list)
                and value
                and all(_is_record(item) for item in value)
            ):
                found.append((node, str(key)))
            else:
                found.extend(_record_lists(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_record_lists(item))
    return found


def collect_records(node: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parent, key in _record_lists(node):
        records.extend(item for item in parent[key] if _is_record(item))
    return records


def slim_record(item: dict[str, Any], *, summary_chars: int = _SUMMARY_CHARS) -> dict[str, Any]:
    slim: dict[str, Any] = {}
    title = item.get("title")
    summary = item.get("summary")
    if isinstance(title, str) and title.strip():
        slim["title"] = title.strip()[:_TITLE_CHARS]
    if isinstance(summary, str) and summary.strip():
        slim["summary"] = summary.strip()[:summary_chars]
    for key in ("url", "id", "entity"):
        value = item.get(key)
        if value not in (None, ""):
            slim[key] = value
    return slim or {"title": str(item)[:_TITLE_CHARS]}


def project_tool_json(text: str, *, limit: int, source_seq: int | None) -> str | None:
    """列表按整条记录缩短或丢掉。放不下时不从记录中间切开。"""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict) or not _record_lists(parsed):
        return None

    total = len(collect_records(parsed))
    for parent, key in _record_lists(parsed):
        parent[key] = [slim_record(item) for item in parent[key] if _is_record(item)]
    ref = {"seq": source_seq, "total": total, "omitted": 0}
    parsed["log_ref"] = ref

    omitted = 0
    while len(json.dumps(parsed, ensure_ascii=False)) > limit:
        lists = _record_lists(parsed)
        if not lists:
            break
        parent, key = max(lists, key=lambda item: len(item[0][item[1]]))
        if not parent[key]:
            break
        parent[key].pop()
        omitted += 1
        ref["omitted"] = omitted

    projected = json.dumps(parsed, ensure_ascii=False)
    if len(projected) > limit:
        return None
    return projected
