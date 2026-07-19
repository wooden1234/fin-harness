"""PDF Agent 检索节点。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from retrieval import get_pdf_retriever
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


async def retrieve_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    del config
    query = str(state.get("query") or state.get("original_query") or "").strip()
    retriever = get_pdf_retriever(
        top_k=5,
        similarity_threshold=None,
        metadata_filters=None,
        hybrid=True,
    )
    hits = retriever.search(query, top_k=5, metadata_filters=None)
    quality_calibrator = _get_quality_calibrator()
    quality_calibrator.annotate(hits)
    trace = getattr(retriever, "last_trace", None)
    logger.info(
        "retrieval query={} hits={} top1_score={}",
        query[:80],
        len(hits),
        hits[0].score if hits else None,
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
