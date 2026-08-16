"""项目 Skill 加载器：只读取受控目录中的说明文件。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PROJECT_SKILLS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """供 Agent 使用的 Skill 说明。"""

    name: str
    path: str
    instructions: str
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()

    @property
    def tool_ids(self) -> tuple[str, ...]:
        """返回 Skill 声明的全部工具，必需工具排在可选工具之前。"""
        return self.required_tools + self.optional_tools


def _parse_front_matter(text: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """解析 Skill 文件顶部的简化 YAML 元数据，避免引入额外依赖。"""
    if not text.startswith("---\n"):
        return {}, {}
    marker = text.find("\n---", 4)
    if marker < 0:
        raise ValueError("skill_front_matter_not_closed")

    scalars: dict[str, str] = {}
    lists: dict[str, tuple[str, ...]] = {}
    current_list: str | None = None
    for raw_line in text[4:marker].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-") and current_list is not None:
            value = line[1:].strip()
            if value:
                lists[current_list] = (*lists.get(current_list, ()), value)
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError("skill_front_matter_invalid")
        key = key.strip()
        value = value.strip()
        if key in {"required_tools", "optional_tools"}:
            current_list = key
            lists[key] = ()
        else:
            current_list = None
            scalars[key] = value
    return scalars, lists


def load_skill(name: str) -> SkillDocument:
    """加载项目 Skill，不执行 Skill 中的脚本。"""
    normalized = name.strip().lower()
    if not _SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError("invalid_skill_name")

    skill_path = (_PROJECT_SKILLS_ROOT / normalized / "SKILL.md").resolve()
    try:
        skill_path.relative_to(_PROJECT_SKILLS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("skill_path_outside_project") from exc
    if not skill_path.is_file():
        raise FileNotFoundError(f"skill_not_found:{normalized}")

    text = skill_path.read_text(encoding="utf-8")
    scalars, lists = _parse_front_matter(text)
    declared_name = scalars.get("name")
    if declared_name and declared_name != normalized:
        raise ValueError("skill_name_mismatch")
    return SkillDocument(
        name=normalized,
        path=str(skill_path),
        instructions=text,
        required_tools=lists.get("required_tools", ()),
        optional_tools=lists.get("optional_tools", ()),
    )


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    name: str
    description: str


def list_skill_catalog() -> tuple[SkillCatalogEntry, ...]:
    entries: list[SkillCatalogEntry] = []
    for path in sorted(_PROJECT_SKILLS_ROOT.glob("*/SKILL.md")):
        name = path.parent.name
        if not _SKILL_NAME_RE.fullmatch(name):
            continue
        try:
            document = load_skill(name)
        except (FileNotFoundError, ValueError):
            continue
        description = ""
        for line in document.instructions.splitlines():
            if line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("\"'")
                break
        entries.append(SkillCatalogEntry(name=name, description=description))
    return tuple(entries)


__all__ = ["SkillCatalogEntry", "SkillDocument", "list_skill_catalog", "load_skill"]
