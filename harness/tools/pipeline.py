"""工具管线：超时包装。"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from pydantic import ValidationError

from harness.tools.arguments import coerce_tool_arguments, merge_tool_results
from harness.tools.definition import ToolDefinition
from harness.tools.errors import MALFORMED_ARGUMENTS, error_result


class ToolPipeline:
    async def run(self, definition: ToolDefinition, arguments: dict[str, Any] | str | None) -> Mapping[str, Any]:
        payloads = coerce_tool_arguments(arguments, openai_schema=definition.openai_schema)
        if not payloads:
            return error_result(MALFORMED_ARGUMENTS, tool=definition.tool_id)

        async def _invoke(payload: dict[str, Any]) -> Mapping[str, Any]:
            try:
                return await asyncio.wait_for(
                    definition.handler(payload),
                    timeout=definition.timeout_seconds,
                )
            except asyncio.TimeoutError:
                return error_result("tool_timeout", tool=definition.tool_id)
            except ValidationError as exc:
                return error_result(
                    MALFORMED_ARGUMENTS,
                    tool=definition.tool_id,
                    message=str(exc).split("For further information", 1)[0].strip()[:400],
                )
            except Exception as exc:  # noqa: BLE001
                return error_result(
                    "tool_execution_failed",
                    tool=definition.tool_id,
                    message=str(exc)[:400],
                )

        if len(payloads) == 1:
            return await _invoke(payloads[0])
        gathered = await asyncio.gather(*[_invoke(item) for item in payloads])
        return merge_tool_results(list(gathered))
