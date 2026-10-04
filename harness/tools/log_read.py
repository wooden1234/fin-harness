"""按日志序号读取投影省略的工具结果片段。"""

from __future__ import annotations

import json
from typing import Any

from harness.compaction.project import collect_records, slim_record
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

_MAX_ITEMS = 5
_MAX_CHARS = 2000
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
        "char_offset": {
            "type": "integer",
            "description": "从原文第几个字开始。普通文本用 log_ref.span 的起点",
        },
        "char_limit": {
            "type": "integer",
            "description": "本次读取字数，最多 2000。传入后按原文切片，不再按记录翻页",
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


def _wants_char_slice(arguments: dict[str, Any]) -> bool:
    return any(
        key in arguments and arguments.get(key) is not None
        for key in ("char_offset", "char_limit")
    )


def read_tool_log_definition(store: Any, session_id: str) -> ToolDefinition:
    async def _read(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            source_seq = int(arguments.get("source_seq"))
        except (TypeError, ValueError):
            return error_result("invalid_input", message="source_seq 必须是整数")
        char_slice = _wants_char_slice(arguments)
        if char_slice:
            try:
                char_offset = max(0, int(arguments.get("char_offset") or 0))
                raw_limit = arguments.get("char_limit")
                char_limit = _MAX_CHARS if raw_limit is None else int(raw_limit)
                char_limit = min(_MAX_CHARS, max(1, char_limit))
            except (TypeError, ValueError):
                return error_result("invalid_input", message="char_offset 和 char_limit 必须是整数")
        else:
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
        content = str(event.data.get("content") or "")
        if char_slice:
            chunk = content[char_offset : char_offset + char_limit]
            next_offset = char_offset + len(chunk)
            return {
                "ok": True,
                "source_seq": source_seq,
                "chars": len(content),
                "char_offset": char_offset,
                "content": chunk,
                "next_offset": next_offset if next_offset < len(content) else None,
            }
        records = _load_records(content)
        if records is None:
            return error_result("log_has_no_records", message="这个日志序号里没有可按条读取的记录")
        chunk_records = records[offset : offset + limit]
        return {
            "ok": True,
            "source_seq": source_seq,
            "total": len(records),
            "offset": offset,
            "items": [slim_record(item, summary_chars=480) for item in chunk_records],
        }

    return ToolDefinition(
        tool_id="log.read_tool",
        name="read_tool_log",
        description=(
            "按 log_ref.seq 从会话日志读取被省略的工具结果。"
            "记录列表用 offset、limit 续读，一次最多 5 条。"
            "普通文本或要读记录摘要以外的原文时，用 char_offset、char_limit，一次最多 2000 字；"
            "log_ref.span 是被切掉的区间。不要一次读取全部全文。"
        ),
        handler=_read,
        openai_schema=function_schema(
            "read_tool_log",
            "按日志序号分段读取工具结果原文。有 span 时用 char_offset 读取被切掉的区间",
            _PARAMETERS,
        ),
        is_concurrency_safe=True,
        read_only=True,
    )
