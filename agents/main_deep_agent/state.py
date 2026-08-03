"""Main DeepAgent 的可恢复执行日志。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.orchestrator.contracts import Evidence


_TODO_STATUSES = frozenset({"pending", "in_progress", "completed"})
_MAX_TODOS = 20
_MAX_TODO_CONTENT_LENGTH = 300


def normalize_agent_todos(raw_todos: object) -> list[dict[str, str]]:
    """规范化模型自报计划，限制数量和文本长度以便安全观测。"""
    if not isinstance(raw_todos, list):
        return []
    normalized: list[dict[str, str]] = []
    for raw in raw_todos[:_MAX_TODOS]:
        if not isinstance(raw, Mapping):
            continue
        content = str(raw.get("content") or "").strip()[:_MAX_TODO_CONTENT_LENGTH]
        status = str(raw.get("status") or "").strip()
        if not content or status not in _TODO_STATUSES:
            continue
        normalized.append({"content": content, "status": status})
    return normalized


@dataclass(slots=True)
class MainAgentJournalEntry:
    """一次工具调用的审计摘要。"""

    tool_id: str
    source_family: str
    status: str
    duration_ms: float
    evidence_ids: list[str] = field(default_factory=list)
    error: str = ""
    entity: str = ""
    purpose: str = ""
    expected_fields: list[str] = field(default_factory=list)
    source_preference: str = ""


@dataclass(slots=True)
class MainAgentProgressJournal:
    """工具完成即写入，硬超时后仍可用于确定性恢复。"""

    entries: list[MainAgentJournalEntry] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    agent_todos: list[dict[str, str]] = field(default_factory=list)
    quality_revision_count: int = 0
    quality_trigger_codes: list[str] = field(default_factory=list)
    quality_before_supported_count: int = 0
    quality_after_supported_count: int = 0
    quality_revision_outcome: str = "not_needed"

    def set_agent_todos(self, raw_todos: object) -> None:
        """保存最终 todo 快照；该状态仅用于观测，不作为完成判定。"""
        self.agent_todos = normalize_agent_todos(raw_todos)

    def record(
        self,
        *,
        tool_id: str,
        source_family: str,
        status: str,
        duration_ms: float,
        evidence: list[Evidence],
        error: str = "",
        entity: str = "",
        purpose: str = "",
        expected_fields: list[str] | None = None,
        source_preference: str = "",
    ) -> None:
        for item in evidence:
            self.evidence[item.evidence_id] = item
        self.entries.append(
            MainAgentJournalEntry(
                tool_id=tool_id,
                source_family=source_family,
                status=status,
                duration_ms=duration_ms,
                evidence_ids=[item.evidence_id for item in evidence],
                error=error,
                entity=entity,
                purpose=purpose,
                expected_fields=list(expected_fields or []),
                source_preference=source_preference,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "entries": [
                {
                    "tool_id": item.tool_id,
                    "source_family": item.source_family,
                    "status": item.status,
                    "duration_ms": item.duration_ms,
                    "evidence_ids": item.evidence_ids,
                    "error": item.error,
                    "entity": item.entity,
                    "purpose": item.purpose,
                    "expected_fields": item.expected_fields,
                    "source_preference": item.source_preference,
                }
                for item in self.entries
            ],
            "source_families": sorted(
                {
                    item.source_family
                    for item in self.entries
                    if item.status == "completed"
                    and item.source_family not in {"calculation", "unknown"}
                }
            ),
            "evidence_count": len(self.evidence),
            "agent_todos": [dict(item) for item in self.agent_todos],
            "quality_revision_count": self.quality_revision_count,
            "quality_trigger_codes": list(self.quality_trigger_codes),
            "quality_before_supported_count": self.quality_before_supported_count,
            "quality_after_supported_count": self.quality_after_supported_count,
            "quality_revision_outcome": self.quality_revision_outcome,
        }


__all__ = ["MainAgentProgressJournal", "normalize_agent_todos"]
