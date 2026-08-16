"""工具管线：超时包装。"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from harness.tools.definition import ToolDefinition
from harness.tools.errors import error_result


class ToolPipeline:
    async def run(self, definition: ToolDefinition, arguments: dict[str, Any]) -> Mapping[str, Any]:
        try:
            return await asyncio.wait_for(
                definition.handler(arguments),
                timeout=definition.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return error_result("tool_timeout", tool=definition.tool_id)
        except Exception as exc:  # noqa: BLE001
            return error_result("tool_execution_failed", tool=definition.tool_id, message=str(exc)[:400])
