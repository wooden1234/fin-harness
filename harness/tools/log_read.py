"""按日志序号读取投影省略的工具结果片段。"""

from __future__ import annotations

import json
from typing import Any

from harness.compaction.project import collect_records, slim_record
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

_MAX_ITEMS = 5
_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_seq": {
            "type": "integer",
            "description": "投影里 log_ref.seq，对应该条工具结果的原始日志序号",
        },
        "offset": {
            "type": "integer",
            "description": "从第几条记录开始，从 0 计",
        },
        "limit": {
            "type": "integer",
            "description": "本次读取条数，最多 5 条",
        },
    },
    "required": ["source_seq"],
}


def _load_records(content: str) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    records = collect_records(parsed)
    return records or None


def read_tool_log_definition(store: Any, session_id: str) -> ToolDefinition:
    async def _read(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            source_seq = int(arguments.get("source_seq"))
        except (TypeError, ValueError):
            return error_result("invalid_input", message="source_seq 必须是整数")
        offset = arguments.get("offset") or 0
        limit = arguments.get("limit") or _MAX_ITEMS
        try:
            offset = max(0, int(offset))
            limit = min(_MAX_ITEMS, max(1, int(limit)))
        except (TypeError, ValueError):
            return error_result("invalid_input", message="offset 和 limit 必须是整数")

        events = await store.load_events(session_id)
        event = next(
            (
                item
                for item in events
                if item.seq == source_seq and item.event_type == "tool/result"
            ),
            None,
        )
        if event is None:
            return error_result("log_record_not_found", message="日志里没有这个序号的工具结果")
        records = _load_records(str(event.data.get("content") or ""))
        if records is None:
            return error_result("log_has_no_records", message="这个日志序号里没有可按条读取的记录")
        chunk = records[offset : offset + limit]
        return {
            "ok": True,
            "source_seq": source_seq,
            "total": len(records),
            "offset": offset,
            "items": [slim_record(item, summary_chars=480) for item in chunk],
        }

    return ToolDefinition(
        tool_id="log.read_tool",
        name="read_tool_log",
        description=(
            "按 log_ref.seq 从会话日志读取被投影省略的工具记录。"
            "一次最多 5 条，用 offset 续读，不要一次读取全部全文。"
        ),
        handler=_read,
        openai_schema=function_schema(
            "read_tool_log",
            "按日志序号分段读取工具结果原文",
            _PARAMETERS,
        ),
        is_concurrency_safe=True,
        read_only=True,
    )
