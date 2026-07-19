"""PDF Agent 内部节点状态。"""

from __future__ import annotations

from typing import Any, TypedDict

from retrieval import RetrievalHit


class PdfAgentState(TypedDict, total=False):
    original_query: str
    query: str
    sub_task_id: str
    hits: list[RetrievalHit]
    citations: list[dict[str, Any]]
    citation_indices: list[int]
    context: str
    rewrite_strategy: str
    rewrite_reason: str
    rewrite_query: str
    rewrite_count: int
    trace: Any
    rag_trace: dict[str, Any]
    answer: str
    messages: list[Any]
    next_rewrite_strategy: str
    evidence_evaluation: dict[str, Any]
    evidence_evaluation_status: str
    evidence_route: str
    retrieval_quality: float
    retrieval_quality_source: str
    original_hits: list[RetrievalHit]
    rewrite_hits: list[RetrievalHit]
    context_pipeline_trace: dict[str, Any]
