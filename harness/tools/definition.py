"""工具定义。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Mapping

ToolHandler = Callable[[dict[str, Any]], Awaitable[Mapping[str, Any]]]


def function_schema(
    name: str,
    description: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": dict(
                parameters
                or {"type": "object", "additionalProperties": True}
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    handler: ToolHandler
    openai_schema: dict[str, Any]
    is_concurrency_safe: bool = True
    read_only: bool = True
    requires_human_approval: bool = False
    timeout_seconds: float = 30.0
    max_retries: int = 0
