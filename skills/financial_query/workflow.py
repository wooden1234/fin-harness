"""财务查询 Skill，复用现有 financial_query_agent workflow。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent import financial_query_agent
from skills.base import SkillResult


async def run(query: str) -> SkillResult:
    result = await financial_query_agent.ainvoke(
        {
            "messages": [],
            "sub_question": query,
            "financial_query_text": query,
        }
    )
    messages = result.get("messages") or []
    answer = str(messages[-1].content) if messages else ""
    return SkillResult(
        answer=answer,
        citations=list(result.get("citations") or []),
        metadata={k: v for k, v in result.items() if k.startswith("financial_query_")},
    )
