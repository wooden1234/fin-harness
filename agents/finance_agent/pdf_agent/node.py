"""PDF Agent worker：调用内部 LangGraph 并转换为 Finance Agent 结果。"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agents.finance_agent.pdf_agent.evaluation import evaluate_evidence_node
from agents.finance_agent.pdf_agent.generation import answer_node, extract_citation_indices
from agents.finance_agent.pdf_agent.query_rewrite.answer_mismatch import answer_mismatch_node
from agents.finance_agent.pdf_agent.query_rewrite.hyde import hyde_node
from agents.finance_agent.pdf_agent.query_rewrite.step_back import step_back_node
from agents.finance_agent.pdf_agent.retrieval import retrieve_node
from agents.finance_agent.pdf_agent.state import PdfAgentState
from agents.states import Citation, FinAgentState
from app.core.logger import get_logger
from retrieval import RetrievalHit

logger = get_logger(service="pdf_agent")

PDF_NO_CONTEXT_ANSWER = (
    "抱歉，PDF 文档库中暂未找到与您问题直接相关的内容。"
    "您可以尝试补充报告名称、年份、政策主题或关键词，或联系人工客服。"
)
PDF_BUSY_ANSWER = "抱歉，PDF 文档问答服务暂时繁忙，请稍后重试。"


def _choose_strategy(query: str) -> str:
    abstract_markers = ("为什么", "为何", "原因", "如何", "影响", "机制", "趋势")
    return "hyde" if any(marker in str(query or "") for marker in abstract_markers) else "step_back"


def select_rewrite_node(state: PdfAgentState) -> PdfAgentState:
    strategy = str(state.get("next_rewrite_strategy") or "").strip().lower()
    if strategy not in {"step_back", "hyde", "answer_mismatch"}:
        strategy = _choose_strategy(state.get("original_query") or state.get("query") or "")
    current = dict(state.get("rag_trace") or {})
    stages = list(current.get("stages") or [])
    stages.append({"node": "select_rewrite", "status": "ok", "strategy": strategy})
    current["stages"] = stages
    return {"rewrite_strategy": strategy, "rag_trace": current}


def route_after_select_rewrite(state: PdfAgentState) -> str:
    return str(state.get("rewrite_strategy") or "step_back")


def route_after_evaluation(state: PdfAgentState) -> str:
    route = str(state.get("evidence_route") or "web_search")
    if route == "answer":
        return "answer"
    if route in {"rewrite", "answer_mismatch"} and int(state.get("rewrite_count") or 0) < 1:
        return "select_rewrite"
    return "end"


def build_pdf_agent_graph():
    builder = StateGraph(PdfAgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("select_rewrite", select_rewrite_node)
    builder.add_node("step_back", step_back_node)
    builder.add_node("hyde", hyde_node)
    builder.add_node("answer_mismatch", answer_mismatch_node)
    builder.add_node("answer", answer_node)
    builder.add_node("evidence_evaluate", evaluate_evidence_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evidence_evaluate")
    builder.add_conditional_edges(
        "select_rewrite",
        route_after_select_rewrite,
        {"step_back": "step_back", "hyde": "hyde", "answer_mismatch": "answer_mismatch"},
    )
    for node_name in ("step_back", "hyde", "answer_mismatch"):
        builder.add_edge(node_name, "retrieve")
    builder.add_conditional_edges(
        "evidence_evaluate",
        route_after_evaluation,
        {"select_rewrite": "select_rewrite", "answer": "answer", "end": END},
    )
    builder.add_edge("answer", END)
    return builder.compile()


@lru_cache(maxsize=1)
def get_pdf_agent_graph():
    return build_pdf_agent_graph()


def _latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    raise ValueError("无用户消息")


def _build_context(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        source = meta.get("source", "unknown")
        category = meta.get("category", hit.category or "")
        section = meta.get("section_path") or meta.get("section", "")
        page = meta.get("page_num") or meta.get("page")
        page_text = f" page={page}" if page is not None else ""
        parts.append(
            f"[{i}] source={source} category={category}{page_text} section={section}\n{hit.text}"
        )
    return "\n\n".join(parts)


def _hits_to_citations(
    hits: list[RetrievalHit],
    *,
    sub_task_id: str = "",
    answer: str = "",
    indices: list[int] | None = None,
) -> list[Citation]:
    selected = (
        list(indices)
        if indices is not None
        else extract_citation_indices(answer, len(hits)) if answer else []
    )
    selected = [index for index in selected if 0 <= index < len(hits)]
    citations: list[Citation] = []
    for index in selected:
        hit = hits[index]
        citation: Citation = {
            "source": hit.metadata.get("source", ""),
            "snippet": (hit.text or "")[:200],
            "source_type": "pdf",
            "sub_task_id": sub_task_id,
        }
        page = hit.metadata.get("page_num") or hit.metadata.get("page")
        if isinstance(page, int):
            citation["page"] = page
        metadata = hit.metadata
        parent_chunk_id = str(metadata.get("parent_chunk_id") or "").strip()
        root_chunk_id = str(metadata.get("root_chunk_id") or "").strip()
        parent_node_id = str(metadata.get("parent_node_id") or parent_chunk_id or root_chunk_id).strip()
        if parent_chunk_id:
            citation["parent_chunk_id"] = parent_chunk_id
        if root_chunk_id:
            citation["root_chunk_id"] = root_chunk_id
        if parent_node_id:
            citation["parent_node_id"] = parent_node_id

        raw_child_ids = metadata.get("evidence_child_ids") or metadata.get("child_chunk_ids")
        if not raw_child_ids and parent_node_id and hit.node_id:
            raw_child_ids = [hit.node_id]
        if isinstance(raw_child_ids, (list, tuple, set)):
            child_ids = list(dict.fromkeys(str(value).strip() for value in raw_child_ids if str(value).strip()))
            if child_ids:
                citation["child_chunk_ids"] = child_ids
                citation["evidence_child_ids"] = child_ids
        if metadata.get("auto_merged") is not None:
            citation["auto_merged"] = bool(metadata["auto_merged"])
        if metadata.get("auto_merge_child_count") is not None:
            citation["auto_merge_child_count"] = int(metadata["auto_merge_child_count"])
        citations.append(citation)
    return citations


def _rewrite_metadata(result: PdfAgentState) -> dict:
    evidence = result.get("evidence_evaluation") or {}
    return {
        "rewrite_strategy": result.get("rewrite_strategy", ""),
        "rewrite_query": result.get("rewrite_query", ""),
        "rewrite_count": int(result.get("rewrite_count") or 0),
        "evidence_evaluation": evidence,
        "evidence_evaluation_status": result.get("evidence_evaluation_status", "unavailable"),
        "evidence_route": result.get("evidence_route", "web_search"),
        "next_rewrite_strategy": result.get("next_rewrite_strategy", "none"),
        "missing_fields": evidence.get("missing_fields", []),
        "unsupported_facts": evidence.get("unsupported_facts", []),
        "strategy_reason": evidence.get("strategy_reason", ""),
        "web_reason": evidence.get("web_reason", ""),
        "retrieval_quality": float(result.get("retrieval_quality") or 0.0),
        "retrieval_quality_source": result.get("retrieval_quality_source", "unknown"),
    }


async def pdf_agent(state: FinAgentState, config: RunnableConfig = None) -> dict:
    sub_task_id = state.get("sub_task_id", "")
    query = state.get("sub_question", "") or _latest_user_query(list(state.get("messages") or []))
    internal: PdfAgentState = {
        "original_query": query,
        "query": query,
        "sub_task_id": sub_task_id,
        "messages": list(state.get("messages") or []),
        "rewrite_count": 0,
    }

    try:
        result = await get_pdf_agent_graph().ainvoke(internal, config=config)
    except Exception:
        logger.exception("pdf_agent graph invoke failed")
        return {
            "messages": [AIMessage(content=PDF_BUSY_ANSWER)],
            "citations": [],
            "task_results": [{
                "sub_task_id": sub_task_id,
                "question": query,
                "type": "pdf",
                "coverage": "partial",
                "context": "",
                "fallback_to_web": True,
                "fallback_reason": "pdf_graph_error",
            }],
            "steps": ["pdf_agent", "pdf_graph_error"],
        }

    hits = list(result.get("hits") or [])
    context = str(result.get("context") or _build_context(hits))
    metadata = _rewrite_metadata(result)
    top_score = float(hits[0].score) if hits else 0.0
    evidence_route = str(result.get("evidence_route") or "web_search")
    answer = str(result.get("answer") or "")
    citations = _hits_to_citations(
        hits,
        sub_task_id=sub_task_id,
        answer=answer,
        indices=result.get("citation_indices"),
    )
    rag_trace = dict(result.get("rag_trace") or {})

    if not hits or evidence_route != "answer":
        if not hits:
            reason = "pdf_no_candidates"
        else:
            reason = "pdf_evidence_route_web" if evidence_route == "web_search" else "pdf_evidence_insufficient"
        rag_trace.update({
            "final_route": "web_search",
            "fallback_reason": reason,
            "citation_count": 0,
            "rewrite_count": int(result.get("rewrite_count") or 0),
        })
        return {
            "messages": [AIMessage(content=PDF_NO_CONTEXT_ANSWER)],
            # 拒答路径未进入 answer 节点，不得透传引用。
            "citations": [],
            "task_results": [{
                "sub_task_id": sub_task_id,
                "question": query,
                "type": "pdf",
                "context": "（未找到相关文档条目）",
                "coverage": "uncovered",
                "confidence": top_score,
                "fallback_to_web": True,
                "fallback_reason": reason,
                "rag_trace": rag_trace,
                **metadata,
            }],
            "rag_trace": rag_trace,
            "steps": ["pdf_agent", "pdf_graph", "pdf_uncovered"],
        }

    rag_trace.update({
        "final_route": "pdf",
        "fallback_reason": "",
        "citation_count": len(citations),
        "rewrite_count": int(result.get("rewrite_count") or 0),
    })
    return {
        "messages": [AIMessage(content=answer)],
        "citations": citations,
        "task_results": [{
            "sub_task_id": sub_task_id,
            "question": query,
            "type": "pdf",
            "context": f"[LLM 回答] {answer}",
            "coverage": "covered",
            "confidence": top_score,
            "rag_trace": rag_trace,
            **metadata,
        }],
        "rag_trace": rag_trace,
        "steps": ["pdf_agent", "pdf_graph", "pdf_answer_evaluated"],
    }
