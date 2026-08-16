"""Agent 执行进度：面向用户的步骤事件，不暴露内部节点名。"""

from __future__ import annotations

import hashlib
from harness.projection.step_detail import sanitize_step_detail


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
