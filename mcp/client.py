"""MCP client：Streamable HTTP（天眼查 / 魔搭托管）。"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx

from mcp.config import resolve_server_config, tyc_mcp_configured, tyc_mcp_timeout_sec
from mcp.schemas import McpRequest, McpResponse

_MCP_PROTOCOL_VERSION = "2024-11-05"
_ACCEPT = "application/json, text/event-stream"


def _parse_sse_payload(raw: str) -> dict[str, Any] | None:
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_rpc_result(body: Any, *, raw_text: str = "") -> dict[str, Any]:
    if isinstance(body, dict):
        if "result" in body or "error" in body:
            return body
        return {"result": body}
    if raw_text:
        sse = _parse_sse_payload(raw_text)
        if sse is not None:
            return sse
    raise ValueError("unexpected MCP response format")


class McpClient:
    """通过 Streamable HTTP 调用远程 MCP Server。"""

    async def call_tool(self, request: McpRequest) -> McpResponse:
        endpoint = resolve_server_config(request.server)
        if endpoint is None:
            if request.server == "tyc-mcp" and not tyc_mcp_configured():
                return McpResponse(
                    ok=False,
                    error=(
                        "TYC MCP not configured: set TYC_MCP_ENABLED=true and "
                        "TYC_MCP_URL / TYC_MCP_TOKEN in .env"
                    ),
                )
            return McpResponse(
                ok=False,
                error=f"MCP server not configured: {request.server}.{request.tool}",
            )

        timeout = httpx.Timeout(
            connect=10.0,
            read=max(30.0, tyc_mcp_timeout_sec()),
            write=10.0,
            pool=10.0,
        )
        headers = {
            "Accept": _ACCEPT,
            "Content-Type": "application/json",
            "Authorization": endpoint["authorization"],
        }

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                session_id = await self._initialize(client, endpoint["url"], headers)
                if session_id:
                    headers = {**headers, "Mcp-Session-Id": session_id}
                result = await self._tools_call(
                    client,
                    endpoint["url"],
                    headers,
                    tool=request.tool,
                    arguments=request.arguments,
                )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            return McpResponse(ok=False, error=f"MCP HTTP {exc.response.status_code}: {detail}")
        except httpx.RequestError as exc:
            return McpResponse(ok=False, error=f"MCP request failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            return McpResponse(ok=False, error=f"MCP call failed: {exc}")

        if "error" in result:
            err = result["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            return McpResponse(ok=False, error=message or "mcp_rpc_error", data=result)

        payload = result.get("result")
        if isinstance(payload, dict) and payload.get("isError"):
            content = payload.get("content") or []
            message = ""
            if content and isinstance(content[0], dict):
                message = str(content[0].get("text") or "")
            return McpResponse(
                ok=False,
                error=message or "mcp_tool_error",
                data=payload,
            )

        return McpResponse(ok=True, data=payload)

    async def _post_rpc(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        *,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], httpx.Response]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": method,
            "params": params or {},
        }
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            body = response.json()
            return _extract_rpc_result(body), response
        return _extract_rpc_result(None, raw_text=response.text), response

    async def _initialize(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
    ) -> str:
        _, response = await self._post_rpc(
            client,
            url,
            headers,
            method="initialize",
            params={
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fin-harness", "version": "0.1.0"},
            },
        )
        return (response.headers.get("Mcp-Session-Id") or "").strip()

    async def _tools_call(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        *,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        result, _ = await self._post_rpc(
            client,
            url,
            headers,
            method="tools/call",
            params={"name": tool, "arguments": arguments},
        )
        return result
