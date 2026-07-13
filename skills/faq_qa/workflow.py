"""FAQ 问答 Skill。"""

from __future__ import annotations

from agents.finance_agent.faq_agent.node import _build_context, _hits_to_citations
from agents.llm import get_faq_llm
from skills.base import SkillResult
from tools.retrieval import faq_search


async def run(query: str) -> SkillResult:
    hits = faq_search(query)
    citations = _hits_to_citations(hits)
    context = _build_context(hits)
    answer = await get_faq_llm().ainvoke(f"请基于以下材料回答：\n{context}\n\n问题：{query}")
    return SkillResult(answer=str(answer.content), citations=list(citations))
