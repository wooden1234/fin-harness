"""Agent 执行进度：面向用户的步骤事件，不暴露内部节点名。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping


def sanitize_step_detail(raw: object) -> dict | None:
    """只保留前端可渲染的展示字段，拒绝透传原始工具 payload。"""
    if not isinstance(raw, Mapping):
        return None
    detail: dict = {}
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


def build_step_event(
    *,
    step_id: str,
    label: str,
    status: str,
    category: str | None = None,
    short_label: str | None = None,
    detail: object | None = None,
) -> dict:
    payload: dict = {
        "type": "step",
        "id": step_id,
        "label": label,
        "status": status,
    }
    if category:
        payload["category"] = category
    if short_label:
        payload["short_label"] = short_label
    cleaned = sanitize_step_detail(detail)
    if cleaned:
        payload["detail"] = cleaned
    return payload


def build_todo_snapshot_event(raw_todos: object) -> dict:
    """构造全量替换事件，稳定 ID 不暴露 todo 原文哈希以外的内部标识。"""
    todos = raw_todos if isinstance(raw_todos, list) else []
    occurrences: dict[str, int] = {}
    payload: list[dict[str, str]] = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        content = str(todo.get("content") or "").strip()
        status = str(todo.get("status") or "pending")
        if not content:
            continue
        occurrences[content] = occurrences.get(content, 0) + 1
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        payload.append(
            {
                "id": f"todo-{digest}-{occurrences[content]}",
                "content": content,
                "status": status,
            }
        )
    return {"type": "todo_snapshot", "todos": payload}
