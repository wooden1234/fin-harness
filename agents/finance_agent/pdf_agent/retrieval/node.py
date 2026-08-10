"""PDF Agent 检索节点。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from retrieval import RetrievalHit, get_pdf_retriever
from retrieval.core.filters import merge_filters
from retrieval.retrievers.pdf_kb_router import get_pdf_kb_router
from retrieval.retrievers.query_filter_extractor import get_query_filter_extractor
from retrieval.retrievers.retrieval_quality import RetrievalQualityCalibrator
from retrieval.services import admit_pdf_hits

from ..state import PdfAgentState
from ..trace import append_trace
from .multi_query_fuse import (
    FUSION_MODE_NONE,
    FUSION_MODE_ORIGINAL_REWRITE_RRF,
    fuse_original_and_rewrite_hits,
)

logger = get_logger(service="pdf_agent.retrieve")

_FUSIBLE_REWRITE_STRATEGIES = frozenset({"step_back", "hyde"})


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


async def _resolve_route_and_filters(query: str) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """规则优先解析路由和过滤条件，每次查询最多调用一次 LLM。"""
    route = await get_pdf_kb_router(min_confidence=0.5).aroute(query)
    if not route.supported:
        return None, {
            "source": route.source,
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
        "source": route.source,
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

    extractor = get_query_filter_extractor()
    extraction = extractor.extract_rules(
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


def _should_fuse_rewrite(state: PdfAgentState) -> bool:
    rewrite_count = int(state.get("rewrite_count") or 0)
    strategy = str(state.get("rewrite_strategy") or "").strip().lower()
    original_hits = list(state.get("original_hits") or [])
    return (
        rewrite_count >= 1
        and strategy in _FUSIBLE_REWRITE_STRATEGIES
        and bool(original_hits)
    )


async def retrieve_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    del config
    query = str(state.get("query") or state.get("original_query") or "").strip()
    original_query = str(state.get("original_query") or query).strip()
    rewrite_count = int(state.get("rewrite_count") or 0)
    strategy = str(state.get("rewrite_strategy") or "").strip().lower()
    prior_original_hits = list(state.get("original_hits") or [])
    quality_calibrator = _get_quality_calibrator()
    fuse_rewrite = _should_fuse_rewrite(state)

    metadata_filters, route_meta, filter_meta = await _resolve_route_and_filters(query)
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
            fusion_mode=FUSION_MODE_NONE,
            original_hit_count=len(prior_original_hits),
            rewrite_hit_count=0,
            fused_hits=0,
        )
        update: PdfAgentState = {
            "hits": [],
            "trace": None,
            "context": "",
            "retrieval_quality": 0.0,
            "retrieval_quality_source": quality_calibrator.source,
            **trace_update,
        }
        if rewrite_count >= 1:
            update["rewrite_hits"] = []
        return update

    top_k = max(int(settings.PDF_RETRIEVAL_TOP_K), 1)
    retriever = get_pdf_retriever(
        top_k=top_k,
        similarity_threshold=None,
        metadata_filters=metadata_filters,
        hybrid=True,
        rerank_min_score=settings.RERANK_MIN_SCORE,
    )

    fusion_mode = FUSION_MODE_NONE
    fused_count = 0
    rewrite_hits: list[RetrievalHit] = []
    original_hits_out: list[RetrievalHit] | None = None

    if fuse_rewrite:
        # 二次检索先关路径内 rerank，融合后再按原问统一 rerank。
        rerank_enabled = bool(getattr(retriever, "rerank_enabled", False))
        retriever.rerank_enabled = False
        try:
            rewrite_hits = await retriever.asearch(
                query, top_k=top_k, metadata_filters=metadata_filters
            )
            rewrite_hits = admit_pdf_hits(rewrite_hits, use_mode="direct_qa")
        finally:
            retriever.rerank_enabled = rerank_enabled

        fuse_result = await fuse_original_and_rewrite_hits(
            original_query=original_query,
            original_hits=prior_original_hits,
            rewrite_hits=rewrite_hits,
            retriever=retriever,
            top_k=top_k,
        )
        hits = admit_pdf_hits(fuse_result.hits, use_mode="direct_qa")
        fusion_mode = fuse_result.fusion_mode
        fused_count = fuse_result.fused_count
        logger.info(
            "retrieval fused strategy={} original={} rewrite={} fused={} final={}",
            strategy,
            len(prior_original_hits),
            len(rewrite_hits),
            fused_count,
            len(hits),
        )
    else:
        hits = await retriever.asearch(query, top_k=top_k, metadata_filters=metadata_filters)
        hits = admit_pdf_hits(hits, use_mode="direct_qa")
        if rewrite_count == 0:
            original_hits_out = list(hits)
        elif rewrite_count >= 1:
            # answer_mismatch 等：覆盖重搜，仅记录 rewrite 路便于观测。
            rewrite_hits = list(hits)

    quality_calibrator.annotate(hits)
    trace = getattr(retriever, "last_trace", None)
    logger.info(
        "retrieval query={} hits={} top1_score={} categories={} filters={} fusion={}",
        query[:80],
        len(hits),
        hits[0].score if hits else None,
        route_meta.get("categories"),
        metadata_filters,
        fusion_mode,
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
        fusion_mode=fusion_mode,
        original_hit_count=(
            len(original_hits_out)
            if original_hits_out is not None
            else len(prior_original_hits)
        ),
        rewrite_hit_count=len(rewrite_hits),
        fused_hits=fused_count if fusion_mode == FUSION_MODE_ORIGINAL_REWRITE_RRF else 0,
        rewrite_strategy=strategy or "none",
    )
    result: PdfAgentState = {
        "hits": hits,
        "trace": trace,
        "context": build_retrieval_context(hits),
        "retrieval_quality": float(hits[0].metadata.get("retrieval_quality") or 0.0)
        if hits
        else 0.0,
        "retrieval_quality_source": quality_calibrator.source,
        **trace_update,
    }
    if original_hits_out is not None:
        result["original_hits"] = original_hits_out
    if rewrite_count >= 1:
        result["rewrite_hits"] = rewrite_hits
    # 顶层便于 Bad Case 对比覆盖 vs 融合
    rag_trace = dict(result.get("rag_trace") or {})
    rag_trace["fusion_mode"] = fusion_mode
    rag_trace["original_hit_count"] = (
        len(original_hits_out)
        if original_hits_out is not None
        else len(prior_original_hits)
    )
    rag_trace["rewrite_hit_count"] = len(rewrite_hits)
    rag_trace["fused_hits"] = (
        fused_count if fusion_mode == FUSION_MODE_ORIGINAL_REWRITE_RRF else 0
    )
    result["rag_trace"] = rag_trace
    return result
