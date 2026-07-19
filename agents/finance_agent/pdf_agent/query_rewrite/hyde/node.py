"""HyDE 查询扩展节点。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_pdf_llm

from ...state import PdfAgentState
from ...trace import append_trace
from .prompt import PDF_HYDE_PROMPT


async def hyde_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    query = str(state.get("original_query") or state.get("query") or "").strip()
    try:
        response = await get_pdf_llm().ainvoke(
            [SystemMessage(content=PDF_HYDE_PROMPT), HumanMessage(content=query)],
            config=config,
        )
        rewritten = response.content if isinstance(response.content, str) else str(response.content)
        rewritten = rewritten.strip()
    except Exception:
        rewritten = ""
    trace_update = append_trace(
        state,
        "hyde",
        status="ok" if rewritten else "fallback",
        rewrite_query=rewritten or query,
        rewrite_count=int(state.get("rewrite_count") or 0) + 1,
    )
    return {
        "query": rewritten or query,
        "rewrite_query": rewritten,
        "rewrite_strategy": "hyde",
        "rewrite_count": int(state.get("rewrite_count") or 0) + 1,
        **trace_update,
    }
