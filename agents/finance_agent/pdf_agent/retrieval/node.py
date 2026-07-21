"""PDF Agent 检索节点。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from retrieval import get_pdf_retriever
from retrieval.core.filters import merge_filters
from retrieval.retrievers.pdf_kb_router import get_pdf_kb_router
from retrieval.retrievers.query_filter_extractor import get_query_filter_extractor
from retrieval.retrievers.retrieval_quality import RetrievalQualityCalibrator

from ..state import PdfAgentState
from ..trace import append_trace

logger = get_logger(service="pdf_agent.retrieve")


@lru_cache(maxsize=1)
def _get_quality_calibrator() -> RetrievalQualityCalibrator:
    path = str(settings.PDF_RETRIEVAL_QUALITY_MODEL_PATH or "").strip()
    if path and Path(path).exists():
        try:
            return RetrievalQualityCalibrator.load(path)
        except Exception as exc:
            logger.warning("retrieval quality model load failed error={}", type(exc).__name__)
    return RetrievalQualityCalibrator(source="heuristic")


def build_retrieval_context(hits) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page_num") or metadata.get("page")
        page_text = f" page={page}" if page is not None else ""
        section = metadata.get("section_path") or metadata.get("section", "")
        parts.append(f"[{index}] source={source}{page_text} section={section}\n{hit.text}")
    return "\n\n".join(parts)


def _resolve_route_and_filters(query: str) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """LLM 类别路由 + query_filter_llm，与 gate070 评测链路对齐。"""
    route = get_pdf_kb_router(min_confidence=0.5).route(query)
    if not route.supported:
        return None, {
            "source": "llm",
            "supported": False,
            "confidence": route.confidence,
            "reason": route.reason,
            "uncertain": True,
            "fallback_all": False,
            "abstained": True,
            "categories": [],
        }, {"source": "none", "reason": "route_abstained"}

    categories = list(route.categories)
    route_meta = {
        "source": "llm",
        "supported": route.supported,
        "confidence": route.confidence,
        "reason": route.reason,
        "uncertain": route.uncertain,
        "fallback_all": route.fallback_all,
        "abstained": False,
        "categories": categories,
    }

    filters: dict[str, Any] = {}
    if categories:
        filters["category"] = categories[0] if len(categories) == 1 else categories

    extraction = get_query_filter_extractor().extract(
        query,
        knowledge_bases=categories or None,
    )
    if extraction.filters:
        filters = merge_filters(filters, extraction.filters)
        filter_meta = {
            "source": "llm" if extraction.used_llm else "rules",
            "reason": extraction.reason,
        }
    else:
        filter_meta = {"source": "none", "reason": extraction.reason}

    return (filters or None), route_meta, filter_meta


async def retrieve_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    del config
    query = str(state.get("query") or state.get("original_query") or "").strip()
    quality_calibrator = _get_quality_calibrator()

    metadata_filters, route_meta, filter_meta = _resolve_route_and_filters(query)
    if route_meta.get("abstained"):
        logger.info(
            "retrieval abstained query={} reason={}",
            query[:80],
            route_meta.get("reason"),
        )
        trace_update = append_trace(
            state,
            "retrieve",
            status="abstained",
            query=query,
            route_meta=route_meta,
            filter_meta=filter_meta,
            final_hits=0,
        )
        return {
            "hits": [],
            "trace": None,
            "context": "",
            "retrieval_quality": 0.0,
            "retrieval_quality_source": quality_calibrator.source,
            **trace_update,
        }

    top_k = max(int(settings.PDF_RETRIEVAL_TOP_K), 1)
    retriever = get_pdf_retriever(
        top_k=top_k,
        similarity_threshold=None,
        metadata_filters=metadata_filters,
        hybrid=True,
        rerank_min_score=settings.RERANK_MIN_SCORE,
    )
    hits = retriever.search(query, top_k=top_k, metadata_filters=metadata_filters)
    quality_calibrator.annotate(hits)
    trace = getattr(retriever, "last_trace", None)
    logger.info(
        "retrieval query={} hits={} top1_score={} categories={} filters={}",
        query[:80],
        len(hits),
        hits[0].score if hits else None,
        route_meta.get("categories"),
        metadata_filters,
    )
    trace_update = append_trace(
        state,
        "retrieve",
        status="ok" if hits else "empty",
        query=query,
        vector_hits=getattr(trace, "vector_hits", 0),
        lexical_hits=getattr(trace, "lexical_hits", 0),
        final_hits=len(hits),
        top1_score=float(hits[0].score) if hits else 0.0,
        score_type=hits[0].score_type if hits else "none",
        score_source=(hits[0].metadata.get("score_source") if hits else "none"),
        rerank_status=(trace.extra.get("rerank_status") if trace else "unknown"),
        rerank_min_score=settings.RERANK_MIN_SCORE,
        route_meta=route_meta,
        filter_meta=filter_meta,
        metadata_filters=metadata_filters or {},
        retrieval_quality=(hits[0].metadata.get("retrieval_quality") if hits else 0.0),
        retrieval_quality_source=(
            hits[0].metadata.get("retrieval_quality_source") if hits else quality_calibrator.source
        ),
    )
    return {
        "hits": hits,
        "trace": trace,
        "context": build_retrieval_context(hits),
        "retrieval_quality": float(hits[0].metadata.get("retrieval_quality") or 0.0)
        if hits
        else 0.0,
        "retrieval_quality_source": quality_calibrator.source,
        **trace_update,
    }
