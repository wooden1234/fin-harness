"""LLM 流式块与组装结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StreamChunk:
    kind: str
    text: str = ""
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""
    index: int = 0
    usage: Mapping[str, Any] | None = None
    cache_hit_tokens: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallDraft:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AssembledAssistant:
    content: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCallDraft, ...] = ()
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None
    cache_hit_tokens: int | None = None


@dataclass
class StreamAssembler:
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_buffers: dict[int, dict[str, str]] = field(default_factory=dict)
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None
    cache_hit_tokens: int | None = None

    def push(self, chunk: StreamChunk) -> None:
        if chunk.kind == "content":
            self.content.append(chunk.text)
        elif chunk.kind == "reasoning":
            self.reasoning.append(chunk.text)
        elif chunk.kind == "tool_call_delta":
            buf = self.tool_buffers.setdefault(
                chunk.index, {"call_id": "", "name": "", "arguments": ""}
            )
            if chunk.call_id:
                buf["call_id"] = chunk.call_id
            if chunk.name:
                buf["name"] = chunk.name
            if chunk.arguments_delta:
                buf["arguments"] += chunk.arguments_delta
        elif chunk.kind == "usage":
            self.usage = dict(chunk.usage or {})
            self.cache_hit_tokens = chunk.cache_hit_tokens
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason

    def finalize(self) -> AssembledAssistant:
        calls: list[ToolCallDraft] = []
        for index in sorted(self.tool_buffers):
            buf = self.tool_buffers[index]
            calls.append(
                ToolCallDraft(
                    call_id=buf["call_id"] or f"call-{index}",
                    name=buf["name"],
                    arguments=buf["arguments"],
                )
            )
        return AssembledAssistant(
            content="".join(self.content),
            reasoning="".join(self.reasoning),
            tool_calls=tuple(calls),
            usage=self.usage,
            finish_reason=self.finish_reason,
            cache_hit_tokens=self.cache_hit_tokens,
        )
