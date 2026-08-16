"""Session 事件投影为现网 SSE。"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from harness.projection.step_detail import detail_from_tool_call, detail_from_tool_result
from harness.session.types import SessionEvent

_FAMILY_LABELS = {
    "weather": "天气数据",
    "market": "金融查询",
    "web": "联网资料",
    "research": "研报与公告",
    "financial": "财务数据",
    "knowledge": "知识库",
    "calculation": "受限计算",
    "skill": "技能说明",
    "answer": "整理答案",
}

_HIDDEN_TOOLS = frozenset({"todo_write"})


def tool_family(name: str) -> str:
    lowered = (name or "").lower()
    if "iwencai" in lowered or "finance-query" in lowered:
        return "market"
    if "weather" in lowered:
        return "weather"
    if lowered.startswith("web") or "search_web" in lowered:
        return "web"
    if "pdf" in lowered or "research" in lowered:
        return "research"
    if "fact" in lowered or "financial" in lowered:
        return "financial"
    if "faq" in lowered or lowered.startswith("knowledge"):
        return "knowledge"
    if "calculat" in lowered:
        return "calculation"
    if lowered == "skill":
        return "skill"
    if lowered == "submit_answer":
        return "answer"
    return "tool"


def sse_cursor_after_completed_turns(events: Iterable[SessionEvent]) -> int:
    """SSE 从上一轮 ``turn/end`` 之后开始，避免把已发布答案重放进新气泡。"""
    cursor = 0
    for event in events:
        if event.event_type == "turn/end":
            cursor = event.seq
    return cursor


def project_session_event(event: SessionEvent) -> list[dict[str, Any]]:
    if event.event_type == "tool/call":
        name = str(event.data.get("name") or "")
        if name in _HIDDEN_TOOLS:
            return []
        family = tool_family(name)
        label = _FAMILY_LABELS.get(family, "资料")
        call_id = str(event.data.get("call_id") or event.seq)
        payload: dict[str, Any] = {
            "type": "step",
            "id": call_id,
            "label": f"正在查询{label}…" if family != "answer" else "资料已就绪，正在整理答案…",
            "status": "running",
            "category": family,
            "short_label": label,
        }
        detail = detail_from_tool_call(event.data.get("arguments"))
        if detail:
            payload["detail"] = detail
        return [payload]
    if event.event_type == "tool/result":
        name = str(event.data.get("name") or "")
        if name in _HIDDEN_TOOLS:
            return []
        family = tool_family(name)
        label = _FAMILY_LABELS.get(family, "资料")
        call_id = str(event.data.get("call_id") or event.seq)
        ok = bool(event.data.get("ok", True))
        payload = {
            "type": "step",
            "id": call_id,
            "label": label if ok else f"查询{label}未取得有效结果",
            "status": "done" if ok else "error",
            "category": family,
            "short_label": label,
        }
        detail = detail_from_tool_result(
            name=name,
            content=event.data.get("content"),
            ok=ok,
        )
        if detail:
            payload["detail"] = detail
        return [payload]
    if event.event_type == "todo/write":
        return [todo_snapshot_event(event.data.get("todos") or [])]
    if event.event_type == "approval/asked":
        name = str(event.data.get("name") or "工具")
        return [
            {
                "type": "interrupt",
                "conversation_id": "",
                "message": f"需要批准调用 {name}。输入同意以继续，或输入拒绝以取消。",
                "approval_id": event.data.get("approval_id"),
                "name": name,
            }
        ]
    if event.event_type == "answer/published":
        markdown = str(event.data.get("markdown") or event.data.get("content") or "")
        if not markdown:
            return []
        return [{"type": "token", "content": markdown}]
    if event.event_type == "turn/end":
        reason = str(event.data.get("reason") or "")
        if reason == "waiting_approval":
            return []
        if reason in {"error", "cancelled", "rejected", "persist_failed", "max_tokens"}:
            return [{"type": "error", "message": "本轮未能发布回答（" + reason + "）。"}]
        return []
    return []


def todo_snapshot_event(raw_todos: Iterable[Any]) -> dict[str, Any]:
    payload: list[dict[str, str]] = []
    occurrences: dict[str, int] = {}
    for item in raw_todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "pending")
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
