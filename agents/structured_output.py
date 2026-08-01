"""兼容 OpenAI 风格 JSON mode 的统一结构化输出入口。"""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
from typing import Any, TypeVar, cast

from langchain_core.runnables import RunnableConfig

StructuredT = TypeVar("StructuredT")

_JSON_MODE_INSTRUCTION = (
    "请严格返回一个合法 JSON object，并满足指定结构；不要输出 Markdown 或额外文本。"
)


def ensure_json_mode_instruction(messages: Sequence[Any]) -> list[Any]:
    """确保请求文本显式包含 JSON，兼容严格校验该前置条件的上游。"""
    copied = list(messages)
    prompt_text = " ".join(_message_text(item) for item in copied).lower()
    if "json" in prompt_text:
        return copied
    return [("system", _JSON_MODE_INSTRUCTION), *copied]


async def ainvoke_json_output(
    model: Any,
    schema: type[StructuredT],
    messages: Sequence[Any],
    *,
    config: RunnableConfig | None = None,
) -> StructuredT:
    """统一调用 json_mode，并注入供应商要求的 JSON 指令。"""
    structured = model.with_structured_output(schema, method="json_mode")
    safe_messages = ensure_json_mode_instruction(messages)
    try:
        if config is None:
            result = await structured.ainvoke(safe_messages)
        else:
            result = await structured.ainvoke(safe_messages, config=config)
    except Exception as exc:  # noqa: BLE001
        recovered = _recover_parser_json(exc, schema)
        if recovered is None:
            raise
        result = recovered
    if isinstance(result, schema):
        return result
    validator = getattr(schema, "model_validate", None)
    if callable(validator):
        return cast(StructuredT, validator(result))
    return cast(StructuredT, result)


def _recover_parser_json(
    exc: BaseException,
    schema: type[StructuredT],
) -> StructuredT | None:
    """从结构化解析异常携带的合法 JSON 中恢复，避免包装层误判导致降级。"""
    raw = getattr(exc, "llm_output", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        payload = json.loads(text)
        validator = getattr(schema, "model_validate", None)
        if not callable(validator):
            return None
        return cast(StructuredT, validator(payload))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _message_text(message: Any) -> str:
    if isinstance(message, tuple) and len(message) >= 2:
        return str(message[1])
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", message))


__all__ = ["ainvoke_json_output", "ensure_json_mode_instruction"]
