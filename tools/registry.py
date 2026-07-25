"""工具统一注册表：元数据 + LangChain Tool 一并管理。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from tools.base import ToolSpec

_REGISTRY: dict[str, "RegisteredTool"] = {}
_BY_NAME: dict[str, str] = {}  # langchain tool.name → tool_id


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """已注册工具条目。"""

    spec: ToolSpec
    langchain_tool: BaseTool | None = None


def register_tool(
    spec: ToolSpec,
    *,
    langchain_tool: BaseTool | None = None,
) -> RegisteredTool:
    """注册工具：``ToolSpec`` 必填；有 ``langchain_tool`` 时可被 ``bind_tools``。"""
    if langchain_tool is not None:
        _BY_NAME[str(langchain_tool.name)] = spec.tool_id

    entry = RegisteredTool(spec=spec, langchain_tool=langchain_tool)
    _REGISTRY[spec.tool_id] = entry
    return entry


def register_tool_spec(spec: ToolSpec) -> None:
    """兼容旧接口：只登记元数据。若已有条目则保留原 langchain_tool。"""
    existing = _REGISTRY.get(spec.tool_id)
    register_tool(
        spec,
        langchain_tool=existing.langchain_tool if existing else None,
    )


def get_registered_tool(tool_id: str) -> RegisteredTool:
    try:
        return _REGISTRY[tool_id]
    except KeyError as exc:
        raise KeyError(f"tool not registered: {tool_id}") from exc


def get_tool_spec(tool_id: str) -> ToolSpec:
    return get_registered_tool(tool_id).spec


def get_langchain_tool(tool_id: str) -> BaseTool:
    entry = get_registered_tool(tool_id)
    if entry.langchain_tool is None:
        raise KeyError(f"tool has no langchain binding: {tool_id}")
    return entry.langchain_tool


def get_tool_id_by_name(name: str) -> str:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"langchain tool name not registered: {name}") from exc


def list_tool_specs() -> list[ToolSpec]:
    return [entry.spec for _, entry in sorted(_REGISTRY.items())]


def list_registered_tools() -> list[RegisteredTool]:
    return [entry for _, entry in sorted(_REGISTRY.items())]


def list_bindable_tools(
    *,
    tool_ids: list[str] | None = None,
    read_only_only: bool = False,
) -> list[BaseTool]:
    """返回可交给 ``llm.bind_tools([...])`` 的工具列表。"""
    wanted = set(tool_ids) if tool_ids is not None else None
    tools: list[BaseTool] = []
    for tool_id, entry in sorted(_REGISTRY.items()):
        if wanted is not None and tool_id not in wanted:
            continue
        if entry.langchain_tool is None:
            continue
        if read_only_only and not entry.spec.read_only:
            continue
        tools.append(entry.langchain_tool)
    return tools


def clear_registry() -> None:
    """仅供测试重置。"""
    _REGISTRY.clear()
    _BY_NAME.clear()


__all__ = [
    "RegisteredTool",
    "clear_registry",
    "get_langchain_tool",
    "get_registered_tool",
    "get_tool_id_by_name",
    "get_tool_spec",
    "list_bindable_tools",
    "list_registered_tools",
    "list_tool_specs",
    "register_tool",
    "register_tool_spec",
]
