"""工具注册表。"""

from __future__ import annotations

from tools.base import ToolSpec

_TOOL_SPECS: dict[str, ToolSpec] = {}


def register_tool_spec(spec: ToolSpec) -> None:
    _TOOL_SPECS[spec.tool_id] = spec


def get_tool_spec(tool_id: str) -> ToolSpec:
    return _TOOL_SPECS[tool_id]


def list_tool_specs() -> list[ToolSpec]:
    return [spec for _, spec in sorted(_TOOL_SPECS.items())]
