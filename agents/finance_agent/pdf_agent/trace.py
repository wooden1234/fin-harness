"""PDF RAG 统一结构化 Trace。"""

from __future__ import annotations

from typing import Any

from .state import PdfAgentState


def append_trace(
    state: PdfAgentState,
    node: str,
    status: str = "ok",
    **details: Any,
) -> dict[str, Any]:
    current = dict(state.get("rag_trace") or {})
    current.setdefault("schema_version", "pdf_rag_trace_v1")
    current.setdefault("query", state.get("original_query") or state.get("query") or "")
    stages = list(current.get("stages") or [])
    stages.append({"node": node, "status": status, **details})
    current["stages"] = stages
    current["rewrite_count"] = int(state.get("rewrite_count") or current.get("rewrite_count") or 0)
    return {"rag_trace": current}

