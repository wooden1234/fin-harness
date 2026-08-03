"""把 MCP Gateway 能力注册为受控本地 Tool。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from mcp.gateway import McpGateway
from tools.core.base import ToolRiskLevel, ToolSpec
from tools.core.registry import RegisteredTool, register_tool


def register_mcp_tool(
    *,
    tool_id: str,
    name: str,
    description: str,
    server: str,
    remote_tool: str,
    gateway: McpGateway | None = None,
    risk_level: ToolRiskLevel = "medium",
    read_only: bool = True,
    timeout_seconds: float = 30.0,
) -> RegisteredTool:
    """将 MCP server.tool 包装为统一 Tool 注册项。"""
    client = gateway or McpGateway()

    async def _call(arguments: dict[str, Any] | None = None) -> Any:
        response = await client.call_tool(
            server=server,
            tool=remote_tool,
            arguments=arguments or {},
        )
        if not response.ok:
            return {
                "ok": False,
                "error": response.error or "mcp_call_failed",
                "data": response.data,
            }
        return response.data

    async def _ainvoke(**kwargs: Any) -> Any:
        return await _call(dict(kwargs))

    langchain_tool = StructuredTool.from_function(
        coroutine=_ainvoke,
        name=name,
        description=description,
    )
    return register_tool(
        ToolSpec(
            tool_id=tool_id,
            name=name,
            description=description,
            risk_level=risk_level,
            read_only=read_only,
            timeout_seconds=timeout_seconds,
        ),
        langchain_tool=langchain_tool,
        handler=_call,
        source="mcp",
    )


__all__ = ["register_mcp_tool"]
