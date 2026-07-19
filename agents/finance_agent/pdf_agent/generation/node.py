"""PDF Agent 答案生成节点。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, SystemMessage

from agents.llm import get_pdf_llm

from .prompt import PDF_GENERATION_PROMPT
from ..state import PdfAgentState
from ..trace import append_trace


def extract_citation_indices(answer: str, hit_count: int) -> list[int]:
    """提取答案实际引用的片段编号，并丢弃越界或重复编号。"""
    indices: list[int] = []
    for raw in re.findall(r"\[(\d+)\]", answer or ""):
        index = int(raw) - 1
        if 0 <= index < hit_count and index not in indices:
            indices.append(index)
    return indices


async def answer_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    question = str(state.get("original_query") or state.get("query") or "").strip()
    llm_messages = [
        SystemMessage(
            content=PDF_GENERATION_PROMPT.format(
                question=question,
                context=state.get("context", ""),
            )
        ),
        *(list(state.get("messages") or [])),
    ]
    parts: list[str] = []
    async for chunk in get_pdf_llm().astream(llm_messages, config=config):
        if chunk.content:
            parts.append(chunk.content if isinstance(chunk.content, str) else str(chunk.content))
    answer = "".join(parts)
    citation_indices = extract_citation_indices(answer, len(state.get("hits") or []))
    trace_update = append_trace(
        state,
        "answer",
        status="ok" if answer else "empty",
        answer_chars=len(answer),
    )
    return {
        "answer": answer,
        "citation_indices": citation_indices,
        "messages": [AIMessage(content=answer)],
        **trace_update,
    }
