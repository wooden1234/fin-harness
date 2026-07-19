"""答案与问题不匹配时的查询改写节点。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_pdf_llm

from ...state import PdfAgentState
from ...trace import append_trace
from .prompt import PDF_ANSWER_MISMATCH_PROMPT


async def answer_mismatch_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    question = str(state.get("original_query") or state.get("query") or "").strip()
    context = str(state.get("context") or "").strip()
    prompt = PDF_ANSWER_MISMATCH_PROMPT.format(question=question, context=context)
    try:
        response = await get_pdf_llm().ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)],
            config=config,
        )
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = rewritten.strip()
    except Exception:
        rewritten = ""

    trace_update = append_trace(
        state,
        "answer_mismatch",
        status="ok" if rewritten else "fallback",
        rewrite_query=rewritten or question,
        rewrite_count=int(state.get("rewrite_count") or 0) + 1,
    )
    return {
        "query": rewritten or question,
        "rewrite_query": rewritten,
        "rewrite_strategy": "answer_mismatch",
        "rewrite_reason": "question_context_mismatch",
        "rewrite_count": int(state.get("rewrite_count") or 0) + 1,
        **trace_update,
    }
