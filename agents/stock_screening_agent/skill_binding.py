"""选股 Agent 的 Skill 加载和工具绑定。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from skills.loader import SkillDocument, load_skill


def _merge_unique(groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class SkillBinding:
    """一次 Agent 运行实际加载的 Skill 及其工具集合。"""

    documents: tuple[SkillDocument, ...]
    required_tools: tuple[str, ...]
    optional_tools: tuple[str, ...]

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return self.required_tools + self.optional_tools

    @property
    def skill_paths(self) -> tuple[str, ...]:
        """返回 Deep Agent 需要读取的相对 Skill 目录。"""
        return tuple(f"skills/{document.name}" for document in self.documents)

    @property
    def primary_required_tool(self) -> str:
        if not self.required_tools:
            names = ",".join(document.name for document in self.documents)
            raise ValueError(f"skill_required_tool_missing:{names}")
        return self.required_tools[0]


def resolve_skill_binding(skill_names: Iterable[str]) -> SkillBinding:
    """加载多个 Skill，并合并其必需和可选工具声明。"""
    names = tuple(str(name).strip() for name in skill_names if str(name).strip())
    if not names:
        raise ValueError("skill_binding_empty")
    documents = tuple(load_skill(name) for name in names)
    return SkillBinding(
        documents=documents,
        required_tools=_merge_unique(
            document.required_tools for document in documents
        ),
        optional_tools=_merge_unique(
            document.optional_tools for document in documents
        ),
    )


__all__ = ["SkillBinding", "resolve_skill_binding"]
