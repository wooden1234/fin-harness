"""兼容大模型常见格式问题的 JSON 解析辅助函数。"""

from __future__ import annotations

import json
import re
from typing import Any


def _repair_unescaped_quotes(value: str) -> str:
    """修复字符串值中误用的 ASCII 双引号，保留合法 JSON 结构引号。"""
    result: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if character == '"' and not escaped:
            if not in_string:
                in_string = True
                result.append(character)
            else:
                remainder = value[index + 1 :].lstrip()
                next_character = remainder[:1]
                if not next_character or next_character in {",", ":", "}", "]"}:
                    in_string = False
                    result.append(character)
                else:
                    result.append(r"\"")
        else:
            result.append(character)
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return "".join(result)


def parse_json_payload(text: str) -> Any:
    """解析模型 JSON，兼容前置文本、代码围栏和未转义引号。"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty llm response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for index, character in enumerate(raw):
        if character not in "[{":
            continue
        candidate_text = raw[index:]
        try:
            payload, _ = decoder.raw_decode(candidate_text)
            return payload
        except json.JSONDecodeError as exc:
            last_error = exc
        repaired = _repair_unescaped_quotes(candidate_text)
        if repaired == candidate_text:
            continue
        try:
            payload, _ = decoder.raw_decode(repaired)
            return payload
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return json.loads(raw)
