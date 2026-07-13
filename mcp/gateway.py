"""MCP 统一出口。"""

from __future__ import annotations

from mcp.client import McpClient
from mcp.schemas import McpRequest, McpResponse


class McpGateway:
    """所有 MCP 调用都从这里出去，便于统一鉴权和审计。"""

    def __init__(self, client: McpClient | None = None) -> None:
        self._client = client or McpClient()

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict | None = None,
    ) -> McpResponse:
        return await self._client.call_tool(
            McpRequest(server=server, tool=tool, arguments=arguments or {})
        )
