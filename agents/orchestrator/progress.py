"""专业 Agent 的安全中间态 journal，用于软截止后的结果恢复。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.orchestrator.adapters import (
    agent_result_from_task_result,
    evidence_from_citation,
)
from agents.orchestrator.contracts import AgentResult, TaskSpec

_STATUS_RANK = {
    "completed": 4,
    "partial": 3,
    "clarify": 2,
    "uncovered": 1,
    "failed": 0,
}


@dataclass(slots=True)
class AgentProgressJournal:
    """仅保留可进入 Root 证据链的公开结果，不保存子图私有状态。"""

    task_id: str
    agent_id: str
    last_node: str = ""
    summary: str = ""
    _task_results: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    _agent_results: dict[str, AgentResult] = field(default_factory=dict)
    _citations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def observe_finance_state(self, state: Mapping[str, Any]) -> None:
        """从 Finance values stream 投影可恢复字段。"""
        for raw in list(state.get("task_results") or []):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            key = (
                str(item.get("sub_task_id") or ""),
                str(item.get("type") or ""),
            )
            current = self._task_results.get(key)
            if current is None or self._coverage_rank(item) >= self._coverage_rank(
                current
            ):
                self._task_results[key] = item

        for raw in list(state.get("agent_results") or []):
            try:
                item = (
                    raw
                    if isinstance(raw, AgentResult)
                    else AgentResult.model_validate(raw)
                )
            except (TypeError, ValueError):
                continue
            key = item.logical_task_id or item.task_id
            current = self._agent_results.get(key)
            if current is None or _STATUS_RANK.get(
                item.status, 0
            ) >= _STATUS_RANK.get(current.status, 0):
                self._agent_results[key] = item

        for raw in list(state.get("citations") or []):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            key = str(
                item.get("evidence_id")
                or item.get("url")
                or (
                    str(item.get("source") or ""),
                    str(item.get("sub_task_id") or ""),
                )
            )
            self._citations[key] = item

        summary = str(state.get("summary") or "").strip()
        if summary:
            self.summary = summary
        steps = list(state.get("steps") or [])
        if steps:
            self.last_node = str(steps[-1])

    def salvage(self, task: TaskSpec) -> AgentResult | None:
        """只恢复已有来源证据的结果；无证据内容继续按超时失败处理。"""
        candidates = list(self._agent_results.values())
        candidates.extend(
            agent_result_from_task_result(item, agent_id=task.agent_id)
            for item in self._task_results.values()
        )
        usable = [
            item
            for item in candidates
            if item.status in {"completed", "partial"} and item.evidence
        ]
        if not usable:
            return None
        best = max(
            usable,
            key=lambda item: (
                _STATUS_RANK.get(item.status, 0),
                len(item.evidence),
                bool(item.answer),
            ),
        )
        evidence_by_id = {item.evidence_id: item for item in best.evidence}
        for citation in self._citations.values():
            evidence = evidence_from_citation(citation, task_id=task.task_id)
            evidence_by_id.setdefault(evidence.evidence_id, evidence)
        return best.model_copy(
            update={
                "task_id": task.task_id,
                "logical_task_id": task.logical_task_id or task.task_id,
                "agent_id": task.agent_id,
                "status": "partial",
                "answer": self.summary or best.answer,
                "evidence": list(evidence_by_id.values()),
                "error_code": "run_soft_deadline_salvaged",
                "error_action": None,
                "gaps": [],
                "metadata": {
                    **dict(best.metadata),
                    "salvaged": True,
                    "salvage_stage": self.last_node or "finance_progress",
                },
            }
        )

    @staticmethod
    def _coverage_rank(item: Mapping[str, Any]) -> int:
        coverage = str(item.get("coverage") or "uncovered")
        return {
            "covered": 4,
            "partial": 3,
            "clarify": 2,
            "uncovered": 1,
        }.get(coverage, 0)


__all__ = ["AgentProgressJournal"]
