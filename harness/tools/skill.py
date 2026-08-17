"""skill 工具。"""

from __future__ import annotations

from typing import Any

from skills.loader import load_skill

from harness.session.types import EventDraft
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

SKILL_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "description": "skills/ 下的 skill 名"},
    },
    "required": ["name"],
}


async def execute_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments.get("name") or "").strip()
    if not name:
        return error_result("invalid_skill_name")
    try:
        document = load_skill(name)
    except FileNotFoundError:
        return error_result("skill_not_found", skill=name)
    except ValueError as exc:
        return error_result(str(exc) or "invalid_skill_name", skill=name)
    return {
        "ok": True,
        "skill": document.name,
        "path": document.path,
        "required_tools": list(document.required_tools),
        "additional_contexts": [{"source": "skill", "content": document.instructions}],
    }


def skill_definition() -> ToolDefinition:
    return ToolDefinition(
        tool_id="skill",
        name="skill",
        description="读取项目 skills/ 中的 SKILL.md。需要外部事实时先调用。",
        handler=execute_skill,
        openai_schema=function_schema("skill", "读取项目 skill 说明", SKILL_PARAMETERS),
        is_concurrency_safe=True,
        read_only=True,
        timeout_seconds=2.0,
    )


async def inject_skill_context(store: Any, session_id: str, result: dict[str, Any], *, turn: int, run_id: str) -> None:
    for item in result.get("additional_contexts") or ():
        content = str(item.get("content") or "")
        if not content:
            continue
        await store.append(
            session_id,
            EventDraft(
                event_type="user/message",
                turn=turn,
                run_id=run_id,
                surface_op="append",
                data={"content": content, "source": "skill"},
            ),
        )
