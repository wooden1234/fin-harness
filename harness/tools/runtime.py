"""工具运行时。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from harness.finalization.submit import SUBMIT_ANSWER_TOOL, execute_submit_answer
from harness.tools.definition import ToolDefinition
from harness.tools.errors import MALFORMED_ARGUMENTS, error_result
from harness.tools.pipeline import ToolPipeline
from harness.tools.providers import product_definitions
from harness.tools.scheduler import ToolResolver
from harness.tools.skill import skill_definition


async def _execute_submit_answer(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return error_result(MALFORMED_ARGUMENTS)
    return {"ok": False, "error": "submit_unbound"}


def submit_answer_definition() -> ToolDefinition:
    return ToolDefinition(
        tool_id="submit_answer",
        name="submit_answer",
        description="提交本轮对用户可见的最终回答。每一轮必须调用。",
        handler=_execute_submit_answer,
        openai_schema=dict(SUBMIT_ANSWER_TOOL),
        is_concurrency_safe=False,
        read_only=False,
        timeout_seconds=5.0,
    )


class ToolRuntime(ToolResolver):
    def __init__(
        self,
        definitions: Iterable[ToolDefinition],
        *,
        pipeline: ToolPipeline | None = None,
    ) -> None:
        self._definitions = tuple(definitions)
        self._by_name: dict[str, ToolDefinition] = {}
        self._by_id: dict[str, ToolDefinition] = {}
        for item in self._definitions:
            self._by_name[item.name] = item
            self._by_id[item.tool_id] = item
        self.pipeline = pipeline or ToolPipeline()

    @classmethod
    def builtin(cls) -> "ToolRuntime":
        return cls((skill_definition(), submit_answer_definition()))

    @classmethod
    def product(cls) -> "ToolRuntime":
        return cls((skill_definition(), submit_answer_definition(), *product_definitions()))

    def rebind_submit(self, handler) -> "ToolRuntime":
        definitions = tuple(
            replace(item, handler=handler) if item.tool_id == "submit_answer" else item
            for item in self._definitions
        )
        return ToolRuntime(definitions, pipeline=self.pipeline)

    def with_extra(self, extra: Iterable[ToolDefinition]) -> "ToolRuntime":
        return ToolRuntime((*self._definitions, *extra), pipeline=self.pipeline)

    def resolve(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name) or self._by_id.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [dict(item.openai_schema) for item in self._definitions]
