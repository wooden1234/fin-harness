"""FAQ Agent 节点：Retriever → context → LLM"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.context import conversation_messages
from agents.llm import get_faq_llm
from agents.finance_agent.faq_agent.prompts import FAQ_BUSY_ANSWER, FAQ_SYSTEM_PROMPT
from agents.states import Citation, FinAgentState
from app.core.config import settings
from app.core.logger import get_logger
from retrieval import RetrievalHit, get_faq_retriever

logger = get_logger(service="faq_agent")


def _latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    raise ValueError("无用户消息")


def _build_context(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        src = h.metadata.get("source", "unknown")
        sec = h.metadata.get("section", "")
        parts.append(f"[{i}] source={src} section={sec}\n{h.text}")
    return "\n\n".join(parts)


def _hits_to_citations(hits: list[RetrievalHit], *, sub_task_id: str = "") -> list[Citation]:
    return [
        {
            "source": h.metadata.get("source", ""),
            "snippet": (h.text or "")[:200],
            "source_type": "faq",
            "sub_task_id": sub_task_id,
        }
        for h in hits
    ]


async def faq_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    sub_question = state.get("sub_question", "")
    sub_task_id = state.get("sub_task_id", "")

    if sub_question:
        query = sub_question
    else:
        query = _latest_user_query(list(state.get("messages") or []))

    logger.info("faq_agent query={} sub_task_id={}", query[:80], sub_task_id)

    retriever = get_faq_retriever(top_k=3, similarity_threshold=None)
    hits = retriever.search(query, top_k=3)

    citations = _hits_to_citations(hits, sub_task_id=sub_task_id) if hits else []

    min_score = settings.FAQ_MIN_RELEVANCE_SCORE
    if not hits or hits[0].score < min_score:
        # 未命中或弱相关命中都视为 uncovered：禁止用弱相关片段硬答
        logger.warning("faq_agent no_context hits={} top1_score={}", len(hits), hits[0].score if hits else None)
        return {
            # 拒答/降级：不推 AIMessage，避免「未找到…」流到前端；由 web/summarize 收口。
            "messages": [],
            "citations": [],
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "faq",
                    "context": "（未找到相关知识库条目）",
                    "coverage": "uncovered",
                    "confidence": float(hits[0].score) if hits else 0.0,
                    "fallback_to_web": True,
                    "fallback_reason": "faq_no_context",
                }
            ],
        }

    context = _build_context(hits)
    citations = _hits_to_citations(hits, sub_task_id=sub_task_id)

    llm_messages = [
        SystemMessage(content=FAQ_SYSTEM_PROMPT.format(context=context)),
        *conversation_messages(state),
    ]
    try:
        llm = get_faq_llm()
        parts: list[str] = []
        async for chunk in llm.astream(llm_messages, config=config):
            if chunk.content:
                parts.append(chunk.content if isinstance(chunk.content, str) else str(chunk.content))
        answer = "".join(parts)
    except Exception:
        logger.exception("faq_agent llm invoke failed")
        return {
            "messages": [AIMessage(content=FAQ_BUSY_ANSWER)],
            "citations": citations,
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "faq",
                    "context": context,
                    "coverage": "partial",
                    "confidence": float(hits[0].score),
                }
            ],
        }

    logger.info("faq_agent hits={} top1_score={:.4f}", len(hits), hits[0].score)
    return {
        "messages": [AIMessage(content=answer)],
        "citations": citations,
        "task_results": [
            {
                "sub_task_id": sub_task_id,
                "question": query,
                "type": "faq",
                "context": f"[LLM 回答] {answer}",
                "coverage": "covered",
                "confidence": float(hits[0].score),
            }
        ],
    }
