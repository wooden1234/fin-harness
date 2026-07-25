"""开放式公司研究 Skill 占位。"""

from __future__ import annotations

from skills.base import SkillResult
from tools.web_search import fetch_web_search


async def run(query: str) -> SkillResult:
    web_result = await fetch_web_search(query)
    return SkillResult(
        answer=str(web_result.get("answer") or "研究模块尚未完整配置。"),
        citations=list(web_result.get("citations") or []),
        metadata={"raw_web_result": web_result},
    )
