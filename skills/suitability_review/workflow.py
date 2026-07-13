"""适当性审查 Skill 占位。"""

from __future__ import annotations

from skills.base import SkillResult


async def run(query: str) -> SkillResult:
    return SkillResult(answer="适当性审查模块尚未配置。", metadata={"query": query})
