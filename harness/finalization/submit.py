"""submit_answer 终检。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from compliance.review import review_answer
from harness.session.types import SessionEvent
from harness.tools.definition import function_schema
from harness.tools.errors import error_result

SUBMIT_ANSWER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["direct", "grounded", "clarify"]},
        "heading": {"type": "string"},
        "direct_answer": {"type": "string"},
        "clarification": {"type": "string"},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text"],
            },
        },
        "follow_ups": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["mode"],
}

SUBMIT_ANSWER_TOOL = function_schema(
    "submit_answer",
    "提交本轮对用户可见的最终回答。每一轮必须调用。",
    SUBMIT_ANSWER_PARAMETERS,
)

_DIGIT_RE = re.compile(r"\d")


def collect_evidence(events: Sequence[SessionEvent], *, turn: int) -> set[str]:
    found: set[str] = set()
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        evidence_id = event.data.get("evidence_id")
        if evidence_id:
            found.add(str(evidence_id))
        content = str(event.data.get("content") or "")
        for match in re.findall(r'"evidence_id"\s*:\s*"([^"]+)"', content):
            found.add(match)
    return found


def execute_submit_answer(
    arguments: dict[str, Any],
    *,
    events: Sequence[SessionEvent],
    turn: int,
) -> dict[str, Any]:
    mode = str(arguments.get("mode") or "direct")
    if mode not in {"direct", "grounded", "clarify"}:
        return error_result("invalid_mode")
    if mode == "clarify":
        text = str(arguments.get("clarification") or arguments.get("direct_answer") or "").strip()
        if not text:
            return error_result("empty_clarification")
        markdown = text
    elif mode == "direct":
        text = str(arguments.get("direct_answer") or "").strip()
        if not text:
            return error_result("empty_direct_answer")
        if _DIGIT_RE.search(text) and any(token in text for token in ("元", "营收", "利润", "%", "同比")):
            return error_result("numbers_require_grounded")
        markdown = text
    else:
        statements = arguments.get("statements") or []
        if not isinstance(statements, list) or not statements:
            return error_result("grounded_requires_statements")
        allowed = collect_evidence(events, turn=turn)
        lines: list[str] = []
        heading = str(arguments.get("heading") or "").strip()
        if heading:
            lines.append(f"## {heading}")
        for item in statements:
            if not isinstance(item, dict):
                return error_result("invalid_statement")
            text = str(item.get("text") or "").strip()
            ids = [str(value) for value in (item.get("evidence_ids") or [])]
            if not text:
                return error_result("empty_statement")
            if not ids or any(item_id not in allowed for item_id in ids):
                return error_result("missing_or_unknown_evidence")
            lines.append(text)
        markdown = "\n\n".join(lines)
    decision = review_answer(markdown)
    if getattr(decision, "action", "pass") == "block":
        return error_result("compliance_blocked", reason=str(getattr(decision, "reason", "")))
    follow_ups = [str(item) for item in (arguments.get("follow_ups") or []) if str(item).strip()]
    return {
        "ok": True,
        "published": True,
        "mode": mode,
        "markdown": markdown,
        "follow_ups": follow_ups,
    }
