"""投诉分流 Skill 占位。"""

from __future__ import annotations

from skills.base import SkillResult


async def run(query: str) -> SkillResult:
    return SkillResult(answer="投诉分流模块尚未配置。", metadata={"query": query})
