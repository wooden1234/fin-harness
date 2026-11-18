"""todo_write：整表替换，不上 surface。

本轮第一次调用先请用户确认。同意后写入清单，之后的状态更新直接落盘。
拒绝后本轮不再建立清单。
"""

from __future__ import annotations

from typing import Any, Sequence

from harness.session.types import EventDraft
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

TODO_PLAN_APPROVAL_MESSAGE = (
    "是否建立任务清单？同意会先规划，耗时较长；拒绝则直接完成，精细度略有不足。"
)
TODO_PLAN_DECLINED_GUIDANCE = (
    "用户拒绝建立任务清单。不要再调用 todo_write。直接继续查数并回答。"
)
TODO_PLAN_PENDING_GUIDANCE = "任务清单正在等待用户确认。不要再提交另一份清单。"
TODO_PLAN_REQUIRED_GUIDANCE = (
    "用户要求先确认任务清单。这次查数没有执行。"
    "请立刻单独调用 todo_write，不要同时调用其他工具，等用户同意后再查数。"
)
_PLAN_FIRST_MARKERS = ("任务清单", "先列出", "先规划", "分步完成", "分步骤")


def user_wants_plan_first(events: Sequence[Any], *, turn: int) -> bool:
    """用户本轮明确要求先列清单时，确认前不查数。"""
    for event in reversed(events):
        if getattr(event, "turn", None) != turn:
            continue
        if getattr(event, "event_type", "") != "user/message":
            continue
        data = getattr(event, "data", None) or {}
        if str(data.get("source") or "user") != "user":
            continue
        text = str(data.get("content") or "")
        return any(marker in text for marker in _PLAN_FIRST_MARKERS)
    return False


def todo_plan_state(events: Sequence[Any], *, turn: int) -> str:
    """本轮清单确认状态：ask 尚未问过，open 已写入，declined 用户拒绝。"""
    asked: set[str] = set()
    declined = False
    opened = False
    for event in events:
        if getattr(event, "turn", None) != turn:
            continue
        data = getattr(event, "data", None) or {}
        event_type = getattr(event, "event_type", "")
        if event_type == "approval/asked" and str(data.get("name") or "") == "todo_write":
            approval_id = str(data.get("approval_id") or "")
            if approval_id:
                asked.add(approval_id)
        elif event_type == "approval/decided":
            approval_id = str(data.get("approval_id") or "")
            if approval_id in asked and str(data.get("decision") or "") != "allow":
                declined = True
        elif event_type == "todo/write":
            opened = True
    if opened:
        return "open"
    if declined:
        return "declined"
    return "ask"


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
            "单源数字题跳过；约 3 个及以上数据步骤再调用。"
            "本轮第一次调用必须单独发起，且排在任何查数之前：先等用户确认，同意后再调用其他工具。"
            "拒绝后不要再调用，直接完成任务。用户已同意后的状态更新直接写入。不上模型 surface。"
        ),
        handler=_handle,
        openai_schema=function_schema("todo_write", "更新任务清单", _TODO_PARAMETERS),
        is_concurrency_safe=True,
        read_only=True,
        timeout_seconds=2.0,
    )
