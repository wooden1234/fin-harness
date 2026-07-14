"""PDF Agent 节点：PDF Retriever → context → LLM。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.llm import get_pdf_llm
from agents.finance_agent.pdf_agent.prompts import PDF_BUSY_ANSWER, PDF_NO_CONTEXT_ANSWER, PDF_SYSTEM_PROMPT
from agents.states import Citation, FinAgentState
from app.core.config import settings
from app.core.logger import get_logger
from retrieval import RetrievalHit, get_pdf_retriever
from retrieval.filters import infer_pdf_metadata_filters
from retrieval.pdf_kb_router import get_pdf_kb_router

logger = get_logger(service="pdf_agent")


def _latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    raise ValueError("无用户消息")


def _build_context(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        src = meta.get("source", "unknown")
        category = meta.get("category", h.category or "")
        sec = meta.get("section_path") or meta.get("section", "")
        page = meta.get("page_num") or meta.get("page")
        page_text = f" page={page}" if page is not None else ""
        parts.append(f"[{i}] source={src} category={category}{page_text} section={sec}\n{h.text}")
    return "\n\n".join(parts)


def _hits_to_citations(hits: list[RetrievalHit], *, sub_task_id: str = "") -> list[Citation]:
    citations: list[Citation] = []
    for h in hits:
        citation: Citation = {
            "source": h.metadata.get("source", ""),
            "snippet": (h.text or "")[:200],
            "source_type": "pdf",
            "sub_task_id": sub_task_id,
        }
        page = h.metadata.get("page_num") or h.metadata.get("page")
        if isinstance(page, int):
            citation["page"] = page
        citations.append(citation)
    return citations


async def pdf_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    sub_question = state.get("sub_question", "")
    sub_task_id = state.get("sub_task_id", "")

    if sub_question:
        query = sub_question
    else:
        query = _latest_user_query(list(state.get("messages") or []))

    route = get_pdf_kb_router().route(query)
    categories = list(route.categories) or None
    metadata_filters = infer_pdf_metadata_filters(query, knowledge_bases=categories)
    logger.info(
        "pdf_agent query={} sub_task_id={} routed={} confidence={} filters={}",
        query[:80],
        sub_task_id,
        categories,
        route.confidence,
        metadata_filters,
    )

    retriever = get_pdf_retriever(top_k=5, similarity_threshold=None, metadata_filters=metadata_filters, hybrid=True)
    hits = retriever.search(query, top_k=5, metadata_filters=metadata_filters)
    trace = getattr(retriever, "last_trace", None)
    if trace is not None:
        logger.info(
            "pdf_agent retrieval trace abstained={} reason={} policy={} vector_hits={} lexical_hits={} final_hits={}",
            trace.abstained,
            trace.abstain_reason,
            trace.on_empty_policy,
            trace.vector_hits,
            trace.lexical_hits,
            trace.final_hits,
        )

    citations = _hits_to_citations(hits, sub_task_id=sub_task_id) if hits else []

    min_score = settings.PDF_MIN_RELEVANCE_SCORE
    if not hits or hits[0].score < min_score:
        # 未命中或弱相关命中都视为 uncovered：禁止用弱相关片段硬答
        logger.warning("pdf_agent no_context hits={} top1_score={}", len(hits), hits[0].score if hits else None)
        return {
            "messages": [AIMessage(content=PDF_NO_CONTEXT_ANSWER)],
            "citations": [],
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "pdf",
                    "context": "（未找到相关文档条目）",
                    "coverage": "uncovered",
                    "confidence": float(hits[0].score) if hits else 0.0,
                    "fallback_to_web": True,
                    "fallback_reason": "pdf_no_context",
                }
            ],
            "steps": ["pdf_agent"],
        }

    context = _build_context(hits)
    citations = _hits_to_citations(hits, sub_task_id=sub_task_id)

    history = list(state.get("messages") or [])
    llm_messages = [SystemMessage(content=PDF_SYSTEM_PROMPT.format(context=context)), *history]
    try:
        llm = get_pdf_llm()
        parts: list[str] = []
        async for chunk in llm.astream(llm_messages, config=config):
            if chunk.content:
                parts.append(chunk.content if isinstance(chunk.content, str) else str(chunk.content))
        answer = "".join(parts)
    except Exception:
        logger.exception("pdf_agent llm invoke failed")
        return {
            "messages": [AIMessage(content=PDF_BUSY_ANSWER)],
            "citations": citations,
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "pdf",
                    "context": context,
                    "coverage": "partial",
                    "confidence": float(hits[0].score),
                }
            ],
        }

    return {
        "messages": [AIMessage(content=answer)],
        "citations": citations,
        "task_results": [
            {
                "sub_task_id": sub_task_id,
                "question": query,
                "type": "pdf",
                "context": f"[LLM 回答] {answer}",
                "coverage": "covered",
                "confidence": float(hits[0].score),
            }
        ],
    }
