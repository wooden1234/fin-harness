"""工具运行时。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from harness.tools.definition import ToolDefinition
from harness.tools.pipeline import ToolPipeline
from harness.tools.providers import product_definitions
from harness.tools.scheduler import ToolResolver
from harness.tools.skill import skill_definition


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
        return cls((skill_definition(),))

    @classmethod
    def product(cls) -> "ToolRuntime":
        return cls((skill_definition(), *product_definitions()))

    def with_extra(self, extra: Iterable[ToolDefinition]) -> "ToolRuntime":
        return ToolRuntime((*self._definitions, *extra), pipeline=self.pipeline)

    def replace_handler(self, tool_id: str, handler) -> "ToolRuntime":
        definitions = tuple(
            replace(item, handler=handler) if item.tool_id == tool_id else item
            for item in self._definitions
        )
        return ToolRuntime(definitions, pipeline=self.pipeline)

    def exclude(self, *tool_ids: str) -> "ToolRuntime":
        drop = {str(item) for item in tool_ids}
        definitions = tuple(
            item
            for item in self._definitions
            if item.tool_id not in drop and item.name not in drop
        )
        return ToolRuntime(definitions, pipeline=self.pipeline)

    def resolve(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name) or self._by_id.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [dict(item.openai_schema) for item in self._definitions]
