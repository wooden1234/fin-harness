"""工具元数据和返回结果定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolRiskLevel = Literal["low", "medium", "high", "critical"]
ToolSource = Literal["local", "mcp"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """工具注册元数据。"""

    tool_id: str
    name: str
    description: str
    risk_level: ToolRiskLevel = "low"
    read_only: bool = True
    requires_human_approval: bool = False
    timeout_seconds: float = 30.0
    max_retries: int = 0


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。"""

    tool_id: str
    ok: bool
    data: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
