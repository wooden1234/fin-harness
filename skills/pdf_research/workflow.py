"""PDF 研究 Skill。"""

from __future__ import annotations

from agents.finance_agent.pdf_agent.node import _build_context, _hits_to_citations
from agents.llm import get_pdf_llm
from skills.base import SkillResult
from tools.retrieval import pdf_search


async def run(query: str) -> SkillResult:
    hits = pdf_search(query)
    citations = _hits_to_citations(hits)
    context = _build_context(hits)
    answer = await get_pdf_llm().ainvoke(f"请基于以下文档片段回答：\n{context}\n\n问题：{query}")
    return SkillResult(answer=str(answer.content), citations=list(citations))
