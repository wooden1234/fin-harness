"""工具统一注册表：元数据 + LangChain Tool 一并管理。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from tools.core.base import ToolSource, ToolSpec

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]

_REGISTRY: dict[str, "RegisteredTool"] = {}
_BY_NAME: dict[str, str] = {}  # langchain tool.name → tool_id


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """已注册工具条目。"""

    spec: ToolSpec
    handler: ToolHandler
    langchain_tool: BaseTool | None = None
    source: ToolSource = "local"


def register_tool(
    spec: ToolSpec,
    *,
    langchain_tool: BaseTool | None = None,
    handler: ToolHandler | None = None,
    source: ToolSource = "local",
) -> RegisteredTool:
    """注册工具的模型绑定与统一执行入口。

    对同一 ``tool_id`` 重复注册会覆盖旧条目，便于测试清空注册表后
    ``importlib.reload`` 工具模块。
    """
    if langchain_tool is not None:
        tool_name = str(langchain_tool.name)
        existing_id = _BY_NAME.get(tool_name)
        if existing_id is not None and existing_id != spec.tool_id:
            raise ValueError(f"duplicate tool name: {tool_name}")
        _BY_NAME[tool_name] = spec.tool_id

    resolved_handler = handler
    if resolved_handler is None and langchain_tool is not None:
        ainvoke = getattr(langchain_tool, "ainvoke", None)
        if ainvoke is None:
            raise ValueError(f"tool_missing_handler:{spec.tool_id}")
        resolved_handler = ainvoke  # type: ignore[assignment]
    if resolved_handler is None:
        raise ValueError(f"tool_missing_handler:{spec.tool_id}")

    entry = RegisteredTool(
        spec=spec,
        langchain_tool=langchain_tool,
        handler=resolved_handler,
        source=source,
    )
    _REGISTRY[spec.tool_id] = entry
    return entry


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


def validate_tool_ids(tool_ids: list[str] | tuple[str, ...]) -> None:
    """严格校验 Agent 声明的工具 ID，避免拼写错误静默降级。"""
    for tool_id in tool_ids:
        entry = get_registered_tool(str(tool_id))
        if entry.langchain_tool is None:
            raise ValueError(f"tool has no langchain binding: {tool_id}")


def clear_registry() -> None:
    """仅供测试重置。"""
    _REGISTRY.clear()
    _BY_NAME.clear()


__all__ = [
    "RegisteredTool",
    "ToolHandler",
    "clear_registry",
    "get_langchain_tool",
    "get_registered_tool",
    "get_tool_id_by_name",
    "get_tool_spec",
    "list_bindable_tools",
    "list_registered_tools",
    "list_tool_specs",
    "validate_tool_ids",
    "register_tool",
]
