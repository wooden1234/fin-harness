"""Finance Agent 内部结果与 v2 统一契约之间的边界转换。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.shared import Citation, TaskResult
from agents.orchestrator.contracts import AgentResult, Evidence

_COVERAGE_TO_STATUS = {
    "covered": "completed",
    "partial": "partial",
    "uncovered": "uncovered",
    "clarify": "clarify",
}


def _stable_id(*parts: str) -> str:
    value = "|".join(parts).encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:16]


def evidence_from_citation(
    citation: Mapping[str, Any],
    *,
    task_id: str = "",
) -> Evidence:
    """把现有引用转换为统一证据。"""
    source = str(citation.get("source") or citation.get("title") or "")
    snippet = str(citation.get("snippet") or "")
    evidence_id = str(citation.get("evidence_id") or "") or _stable_id(
        task_id,
        str(citation.get("url") or ""),
        source,
        snippet,
    )
    metadata = {
        key: value
        for key, value in citation.items()
        if key
        not in {
            "source",
            "title",
            "snippet",
            "url",
            "published_at",
            "source_type",
            "sub_task_id",
            "evidence_id",
        }
    }
    return Evidence(
        evidence_id=evidence_id,
        task_id=task_id or str(citation.get("sub_task_id") or ""),
        source_type=str(citation.get("source_type") or "unknown"),
        provider=str(citation.get("provider") or citation.get("source_type") or ""),
        title=source,
        content=snippet,
        url=str(citation["url"]) if citation.get("url") else None,
        published_at=(
            str(citation["published_at"])
            if citation.get("published_at")
            else None
        ),
        observed_at=str(
            citation.get("observed_at")
            or datetime.now(timezone.utc).isoformat()
        ),
        confidence=float(citation.get("confidence") or 0.0),
        metadata=metadata,
    )


def citation_from_evidence(evidence: Evidence) -> Citation:
    """把统一证据转换为前端引用结构。"""
    citation: Citation = {
        "source": evidence.title or evidence.provider,
        "snippet": evidence.content,
        "source_type": evidence.source_type,
        "sub_task_id": evidence.task_id,
    }
    if evidence.url:
        citation["url"] = evidence.url
    if evidence.published_at:
        citation["published_at"] = evidence.published_at
    citation.update(evidence.metadata)
    return citation


def agent_result_from_task_result(
    result: TaskResult,
    *,
    agent_id: str = "finance_agent",
) -> AgentResult:
    """把旧 Worker 结果转换为统一 Agent 输出。"""
    coverage = str(result.get("coverage") or "uncovered")
    status = _COVERAGE_TO_STATUS.get(coverage, "uncovered")
    task_id = str(result.get("sub_task_id") or "")
    return AgentResult(
        task_id=task_id,
        agent_id=agent_id,
        status=status,
        answer=str(result.get("context") or ""),
        evidence=[
            evidence_from_citation(item, task_id=task_id)
            for item in list(result.get("citations") or [])
        ],
        gaps=[str(result["fallback_reason"])]
        if result.get("fallback_reason")
        else [],
        error_code=str(result.get("error_code") or ""),
        metadata={
            "question": str(result.get("question") or ""),
            "worker_type": str(result.get("type") or ""),
            "confidence": result.get("confidence"),
        },
    )


__all__ = [
    "agent_result_from_task_result",
    "citation_from_evidence",
    "evidence_from_citation",
]
