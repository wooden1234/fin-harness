"""测试用脚本化 LLM。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence

from harness.llm.types import StreamChunk, ToolCallDraft


@dataclass
class ScriptedTurn:
    chunks: tuple[StreamChunk, ...]


def content_turn(text: str) -> ScriptedTurn:
    return ScriptedTurn((StreamChunk(kind="content", text=text, finish_reason="stop"),))


def tool_turn(name: str, arguments: str, *, call_id: str = "call-1") -> ScriptedTurn:
    return ScriptedTurn(
        (
            StreamChunk(
                kind="tool_call_delta",
                index=0,
                call_id=call_id,
                name=name,
                arguments_delta=arguments,
                finish_reason="tool_calls",
            ),
        )
    )


class FakeLlmAdapter:
    provider = "fake"
    adapter_defaults: Mapping[str, Any] = {"temperature": 0, "provider": "fake", "model": "fake"}

    def __init__(
        self,
        turns: Sequence[ScriptedTurn] | None = None,
        *,
        summaries: Sequence[str] | None = None,
        structured_deltas: Sequence[Any] | None = None,
    ) -> None:
        self._turns = list(turns or [])
        self._summaries = list(summaries or [])
        self._structured_deltas = list(structured_deltas or [])
        self.requests: list[dict[str, Any]] = []

    def push(self, turn: ScriptedTurn) -> None:
        self._turns.append(turn)

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        abort: Any,
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append({"system": system, "messages": list(messages), "tools": list(tools)})
        if abort.is_set():
            from harness.contracts.errors import LlmError

            raise LlmError("cancelled", code="cancelled")
        if not self._turns:
            yield StreamChunk(kind="content", text="", finish_reason="stop")
            return
        turn = self._turns.pop(0)
        for chunk in turn.chunks:
            yield chunk

    async def complete(self, *, system: str, prompt: str) -> str:
        self.requests.append({"kind": "complete", "system": system, "prompt": prompt})
        if self._summaries:
            return str(self._summaries.pop(0)).strip()
        return "【对话摘要】更早的问答与检索已压缩，关键数字与未完成项见后续原文。"

    async def complete_structured(self, schema: type[Any], prompt: str) -> Any:
        self.requests.append({"kind": "complete_structured", "schema": schema, "prompt": prompt})
        if not self._structured_deltas:
            raise RuntimeError("no structured delta")
        payload = self._structured_deltas.pop(0)
        validator = getattr(schema, "model_validate", None)
        if callable(validator) and not isinstance(payload, schema):
            return validator(payload)
        return payload
