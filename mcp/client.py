"""MCP client 封装占位。"""

from __future__ import annotations

from mcp.schemas import McpRequest, McpResponse


class McpClient:
    """后续替换为真实 MCP SDK client。"""

    async def call_tool(self, request: McpRequest) -> McpResponse:
        return McpResponse(
            ok=False,
            error=f"MCP server not configured: {request.server}.{request.tool}",
        )
