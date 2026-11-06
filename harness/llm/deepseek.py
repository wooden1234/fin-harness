"""DeepSeek / OpenAI 兼容 adapter。历史必须带回 assistant.tool_calls。"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Mapping, Sequence

from harness.contracts.errors import LlmError
from harness.llm.types import StreamChunk
from harness.tools.arguments import coerce_tool_arguments


def _stream_idle_timeout_seconds() -> float:
    from app.core.config import settings

    return max(5.0, float(settings.DEEPSEEK_STREAM_CHUNK_TIMEOUT_SEC or 30.0))


def _request_timeout_seconds() -> float:
    from app.core.config import settings

    return max(5.0, float(settings.DEEPSEEK_TIMEOUT_SEC or 60.0))


_STREAM_END = object()


async def _anext_or_end(iterator: Any) -> Any:
    """在 Task 里吞掉 StopAsyncIteration，避免 wait_for 把它变成 RuntimeError。"""
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _STREAM_END


async def _close_stream(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await asyncio.wait_for(aclose(), timeout=2)
    except Exception:
        return


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
        iterator = llm.astream(lc_messages).__aiter__()
        try:
            while True:
                if abort.is_set():
                    raise LlmError("cancelled", code="cancelled")
                try:
                    chunk = await asyncio.wait_for(
                        _anext_or_end(iterator),
                        _stream_idle_timeout_seconds(),
                    )
                except TimeoutError:
                    raise
                if chunk is _STREAM_END:
                    break
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
            yield StreamChunk(
                kind="usage",
                finish_reason="tool_calls" if saw_tool_call else "stop",
            )
        except LlmError:
            raise
        except TimeoutError as exc:
            raise LlmError("模型响应超时", code="timeout", retryable=True) from exc
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
            elif "timeout" in lowered or "timed out" in lowered:
                code, retryable = "timeout", True
            elif "temporar" in lowered:
                code, retryable = "retryable", True
            raise LlmError(message, code=code, retryable=retryable) from exc
        finally:
            await _close_stream(iterator)

    async def complete(self, *, system: str, prompt: str) -> str:
        """非流式补全，供压缩摘要等旁路使用，不走工具调用。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        from agents.llm import get_faq_llm

        llm = get_faq_llm()
        try:
            result = await asyncio.wait_for(
                llm.ainvoke([SystemMessage(content=system), HumanMessage(content=prompt)]),
                _request_timeout_seconds(),
            )
        except TimeoutError as exc:
            raise LlmError("模型响应超时", code="timeout", retryable=True) from exc
        text = getattr(result, "content", result)
        if isinstance(text, list):
            text = "".join(
                part if isinstance(part, str) else str(getattr(part, "text", part) or "")
                for part in text
            )
        return str(text or "").strip()

    async def complete_structured(self, schema: type[Any], prompt: str) -> Any:
        """结构化抽取，供压缩摘要使用；走 router（温度 0）。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        from agents.llm import get_router_llm
        from agents.structured_output import ainvoke_json_output

        system = (
            "请严格返回一个合法 JSON object，并满足指定结构；不要输出 Markdown 或额外文本。"
            "只写窗口里出现过的事实。不要编造数据或买卖建议。"
        )
        try:
            return await asyncio.wait_for(
                ainvoke_json_output(
                    get_router_llm(),
                    schema,
                    [SystemMessage(content=system), HumanMessage(content=prompt)],
                ),
                _request_timeout_seconds(),
            )
        except TimeoutError as exc:
            raise LlmError("模型响应超时", code="timeout", retryable=True) from exc
