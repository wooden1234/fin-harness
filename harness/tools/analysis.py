"""把 finalign.analyze 绑到当前 turn 的 session log。"""

from __future__ import annotations

from typing import Any, Sequence

from capabilities.analysis import (
    MAX_MATERIAL_CHARS,
    MAX_MATERIALS,
    TOOL_ID,
    TOOL_NAME,
    synthesize_answer,
)
from harness.session.types import SessionEvent

_SKIP_TOOLS = frozenset(
    {
        "skill",
        "todo_write",
        "memory_write",
        "memory_delete",
        TOOL_ID,
        TOOL_NAME,
    }
)


def _truncate(text: str, limit: int = MAX_MATERIAL_CHARS) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 16] + "\n…[truncated]"


def collect_turn_materials(events: Sequence[SessionEvent], *, turn: int) -> list[dict[str, Any]]:
    """从本轮 tool/result 收集分析材料，保留 evidence_id。"""
    items: list[dict[str, Any]] = []
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        name = str(event.data.get("name") or "").strip()
        if not name or name in _SKIP_TOOLS or name.startswith("memory_"):
            continue
        content = _truncate(str(event.data.get("content") or ""))
        if not content:
            continue
        evidence_id = event.data.get("evidence_id")
        items.append(
            {
                "tool": name,
                "ok": bool(event.data.get("ok", True)),
                "evidence_id": str(evidence_id) if evidence_id else None,
                "content": content,
            }
        )
        if len(items) >= MAX_MATERIALS:
            break
    return items


def last_user_question(events: Sequence[SessionEvent], *, turn: int) -> str:
    for event in reversed(tuple(events)):
        if event.turn != turn or event.event_type != "user/message":
            continue
        if str(event.data.get("source") or "user") != "user":
            continue
        text = str(event.data.get("content") or "").strip()
        if text:
            return text
    return ""


def finalign_is_ready() -> bool:
    """本地 finalign 可用于成稿时才把分析工具交给规划模型。"""
    from app.core.config import settings

    if not bool(settings.FINANCE_LLM_DRAFT_ENABLED):
        return False
    from agents.llm import is_finance_llm_available

    return is_finance_llm_available()


def bind_finalign_analyze(store: Any, session_id: str, *, turn: int):
    async def _handle(arguments: dict[str, Any]) -> dict[str, Any]:
        events = await store.load_events(session_id)
        materials = collect_turn_materials(events, turn=turn)
        if not materials:
            materials = arguments.get("materials") or []
        question = str(arguments.get("question") or "").strip()
        if not question:
            question = last_user_question(events, turn=turn)
        return await synthesize_answer(question=question, materials=materials)

    return _handle


__all__ = [
    "TOOL_ID",
    "bind_finalign_analyze",
    "collect_turn_materials",
    "finalign_is_ready",
    "last_user_question",
]
