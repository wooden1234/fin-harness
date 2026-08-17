"""DeepSeek / OpenAI 兼容 adapter。历史必须带回 assistant.tool_calls。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Mapping, Sequence

from harness.contracts.errors import LlmError
from harness.llm.types import StreamChunk
from harness.tools.arguments import coerce_tool_arguments


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and "query" in raw and "_raw" not in raw:
        return raw
    payloads = coerce_tool_arguments(raw)
    if payloads:
        return payloads[0]
    if isinstance(raw, dict):
        return {key: value for key, value in raw.items() if key != "_raw"}
    return {}


def to_langchain_messages(system: str, messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    converted: list[Any] = [SystemMessage(content=system)]
    for message in messages:
        role = message.get("role")
        if role == "user":
            converted.append(HumanMessage(content=str(message.get("content") or "")))
            continue
        if role == "assistant":
            tool_calls = []
            for item in list(message.get("tool_calls") or []):
                if not isinstance(item, dict):
                    continue
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "")
                if not call_id or not name:
                    continue
                tool_calls.append(
                    {
                        "id": call_id,
                        "name": name,
                        "args": _parse_tool_args(item.get("arguments") or item.get("args")),
                        "type": "tool_call",
                    }
                )
            converted.append(
                AIMessage(content=str(message.get("content") or ""), tool_calls=tool_calls)
            )
            continue
        if role == "tool":
            converted.append(
                ToolMessage(
                    content=str(message.get("content") or ""),
                    tool_call_id=str(message.get("call_id") or "unknown"),
                )
            )
    return converted


class DeepSeekAdapter:
    provider = "deepseek"
    adapter_defaults: Mapping[str, Any]

    def __init__(self, *, model: str | None = None) -> None:
        from app.core.config import settings

        self.model = model or settings.DEEPSEEK_MODEL
        self.adapter_defaults = {
            "temperature": 0,
            "provider": self.provider,
            "model": self.model,
        }

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        abort: Any,
    ) -> AsyncIterator[StreamChunk]:
        from agents.llm import get_faq_llm

        lc_messages = to_langchain_messages(system, messages)
        llm = get_faq_llm()
        if tools:
            try:
                llm = llm.bind_tools(tools, parallel_tool_calls=True)
            except TypeError:
                llm = llm.bind_tools(tools)
        saw_tool_call = False
        try:
            async for chunk in llm.astream(lc_messages):
                if abort.is_set():
                    raise LlmError("cancelled", code="cancelled")
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    yield StreamChunk(kind="content", text=text)
                tool_calls = getattr(chunk, "tool_call_chunks", None) or []
                for index, item in enumerate(tool_calls):
                    saw_tool_call = True
                    yield StreamChunk(
                        kind="tool_call_delta",
                        index=index,
                        call_id=item.get("id"),
                        name=item.get("name"),
                        arguments_delta=item.get("args") or "",
                    )
        except LlmError:
            raise
        except Exception as exc:
            from loguru import logger

            logger.exception("llm stream failed")
            message = str(exc)
            code = "unknown"
            retryable = False
            lowered = message.lower()
            if "429" in message or "rate" in lowered:
                code, retryable = "rate_limit", True
            elif "context" in lowered or "maximum context" in lowered:
                code = "context_overflow"
            elif "timeout" in lowered or "temporar" in lowered:
                code, retryable = "retryable", True
            raise LlmError(message, code=code, retryable=retryable) from exc
        yield StreamChunk(
            kind="usage",
            finish_reason="tool_calls" if saw_tool_call else "stop",
        )
