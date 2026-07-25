"""受控工具层。

统一用法：
    from tools import load_all_tools, list_bindable_tools
    load_all_tools()
    llm.bind_tools(list_bindable_tools())
"""

from tools.base import ToolRiskLevel, ToolResult, ToolSpec
from tools.catalog import TOOL_MODULES, load_all_tools
from tools.execution import execute_tool
from tools.registry import (
    RegisteredTool,
    get_langchain_tool,
    get_registered_tool,
    get_tool_id_by_name,
    get_tool_spec,
    list_bindable_tools,
    list_registered_tools,
    list_tool_specs,
    register_tool,
    register_tool_spec,
)

__all__ = [
    "TOOL_MODULES",
    "RegisteredTool",
    "ToolRiskLevel",
    "ToolResult",
    "ToolSpec",
    "execute_tool",
    "get_langchain_tool",
    "get_registered_tool",
    "get_tool_id_by_name",
    "get_tool_spec",
    "list_bindable_tools",
    "list_registered_tools",
    "list_tool_specs",
    "load_all_tools",
    "register_tool",
    "register_tool_spec",
]
