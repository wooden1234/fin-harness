"""新旧任务、结果和引用模型之间的兼容转换。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.shared import Citation, SubTask, TaskResult
from agents.orchestrator.contracts import AgentResult, Evidence, TaskSpec

_KNOWN_TASK_TYPES = {"faq", "pdf", "financial_query", "web_search", "general"}
_STATUS_TO_COVERAGE = {
    "completed": "covered",
    "partial": "partial",
    "uncovered": "uncovered",
    "clarify": "clarify",
    "failed": "uncovered",
}
_COVERAGE_TO_STATUS = {value: key for key, value in _STATUS_TO_COVERAGE.items()}


def _stable_id(*parts: str) -> str:
    value = "|".join(parts).encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:16]


def task_spec_from_subtask(task: SubTask) -> TaskSpec:
    """把旧 Planner 子任务转换为统一任务契约。"""
    capabilities = list(task.evidence_chain or [])
    if not capabilities and task.type:
        capabilities = [str(task.type)]
    return TaskSpec(
        task_id=task.id,
        objective=task.question,
        agent_id="finance_agent",
        required_capabilities=capabilities,
        metadata={"intent": task.intent, "reason": task.reason, "legacy_type": task.type},
    )


def subtask_from_task_spec(task: TaskSpec) -> SubTask:
    """把统一任务契约转换为现有 Planner 子任务。"""
    metadata = task.metadata
    candidate_type = str(metadata.get("legacy_type") or "")
    if candidate_type not in _KNOWN_TASK_TYPES:
        candidate_type = next(
            (item for item in task.required_capabilities if item in _KNOWN_TASK_TYPES),
            "faq",
        )
    return SubTask(
        id=task.task_id,
        question=task.objective,
        intent=str(metadata.get("intent") or ""),
        reason=str(metadata.get("reason") or ""),
        type=candidate_type,
        evidence_chain=list(task.required_capabilities),
    )


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
    """把统一证据转换为前端兼容的引用结构。"""
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
            "legacy_type": str(result.get("type") or ""),
            "confidence": result.get("confidence"),
        },
    )


def task_result_from_agent_result(result: AgentResult) -> TaskResult:
    """把统一 Agent 输出转换为现有 Worker 结果。"""
    coverage = _STATUS_TO_COVERAGE[result.status]
    legacy: TaskResult = {
        "sub_task_id": result.task_id,
        "question": str(result.metadata.get("question") or ""),
        "type": str(result.metadata.get("legacy_type") or result.agent_id),
        "context": result.answer,
        "citations": [citation_from_evidence(item) for item in result.evidence],
        "coverage": coverage,
        "fallback_to_web": coverage == "uncovered",
    }
    if result.gaps:
        legacy["fallback_reason"] = result.gaps[0]
    if result.error_code:
        legacy["rag_trace"] = {"error_code": result.error_code}
    return legacy


__all__ = [
    "agent_result_from_task_result",
    "citation_from_evidence",
    "evidence_from_citation",
    "subtask_from_task_spec",
    "task_result_from_agent_result",
    "task_spec_from_subtask",
]
