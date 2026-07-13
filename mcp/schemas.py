"""MCP 调用数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class McpRequest:
    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class McpResponse:
    ok: bool
    data: Any = None
    error: str = ""
