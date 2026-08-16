"""todo_write：整表替换，不上 surface。"""

from __future__ import annotations

from typing import Any

from harness.session.types import EventDraft
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

_TODO_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                },
                "required": ["content", "status"],
            },
        }
    },
    "required": ["todos"],
}


def _validate_todos(raw: Any) -> list[dict[str, str]] | str:
    if not isinstance(raw, list) or not raw:
        return "invalid_todos"
    seen: set[str] = set()
    items: list[dict[str, str]] = []
    in_progress = 0
    for item in raw:
        if not isinstance(item, dict):
            return "invalid_todos"
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "")
        if not content:
            return "invalid_todo_content"
        if content in seen:
            return "duplicate_todo_content"
        if status not in {"pending", "in_progress", "completed"}:
            return "invalid_todo_status"
        seen.add(content)
        if status == "in_progress":
            in_progress += 1
        items.append({"content": content, "status": status})
    if in_progress > 3:
        return "too_many_in_progress"
    return items


def todo_write_definition(store: Any, session_id: str, *, turn: int, run_id: str) -> ToolDefinition:
    async def _handle(arguments: dict[str, Any]) -> dict[str, Any]:
        parsed = _validate_todos(arguments.get("todos"))
        if isinstance(parsed, str):
            return error_result(parsed)
        counts = {
            "pending": sum(1 for item in parsed if item["status"] == "pending"),
            "inProgress": sum(1 for item in parsed if item["status"] == "in_progress"),
            "completed": sum(1 for item in parsed if item["status"] == "completed"),
        }
        await store.append(
            session_id,
            EventDraft(
                event_type="todo/write",
                turn=turn,
                run_id=run_id,
                data={"todos": parsed},
            ),
        )
        return {
            "ok": True,
            "todos": parsed,
            "counts": counts,
            "content": (
                f"Updated todo list: {counts['pending']} pending, "
                f"{counts['inProgress']} in progress, {counts['completed']} completed."
            ),
        }

    return ToolDefinition(
        tool_id="todo_write",
        name="todo_write",
        description=(
            "记录本轮任务清单。每次提交完整列表（替换而非增量）。"
            "单源数字题跳过；≥3 个数据步骤再用。不上模型 surface。"
        ),
        handler=_handle,
        openai_schema=function_schema("todo_write", "更新任务清单", _TODO_PARAMETERS),
        is_concurrency_safe=True,
        read_only=True,
        timeout_seconds=2.0,
    )
