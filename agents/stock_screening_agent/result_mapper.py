"""选股 Tool 结果到 AgentResult 的收敛。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from agents.orchestrator.contracts import AgentResult, Evidence
from agents.stock_screening_agent.skill_binding import (
    SkillBinding,
    resolve_skill_binding,
)
from agents.stock_screening_agent.spec import (
    STOCK_SCREENING_SPEC,
    StockScreeningSpec,
)

_ROW_KEYS = ("rows", "items", "results", "candidates")
_NESTED_DATA_KEYS = ("data", "result", "payload")


def _extract_rows(value: Any) -> list[Any] | None:
    """从问财结果的常见嵌套结构中提取候选记录。"""
    if not isinstance(value, Mapping):
        return None
    for key in _ROW_KEYS:
        rows = value.get(key)
        if isinstance(rows, (list, tuple)):
            return list(rows)
    for key in _NESTED_DATA_KEYS:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            rows = _extract_rows(nested)
            if rows is not None:
                return rows
    return None


def _has_candidate_rows(tool_result: Mapping[str, Any]) -> bool:
    """只有存在至少一条候选记录时，才视为选股成功。"""
    return bool(_extract_rows(tool_result.get("data")))


def agent_result_from_tool_updates(
    query: str,
    updates: Mapping[str, Any],
    *,
    spec: StockScreeningSpec = STOCK_SCREENING_SPEC,
    binding: SkillBinding | None = None,
) -> AgentResult:
    """把工具循环结果收敛为统一 AgentResult。"""
    active_binding = binding or resolve_skill_binding(spec.skills)
    tool_results = list(updates.get("tool_results") or [])
    required = active_binding.primary_required_tool
    required_results = [
        item for item in tool_results if item.get("tool_id") == required
    ]
    successful = next(
        (
            item
            for item in reversed(required_results)
            if item.get("ok") and _has_candidate_rows(item)
        ),
        None,
    )
    empty_success = next(
        (
            item
            for item in reversed(required_results)
            if item.get("ok") and not _has_candidate_rows(item)
        ),
        None,
    )
    failed = next(
        (item for item in reversed(required_results) if not item.get("ok")),
        None,
    )

    structured_data: dict[str, Any] = {}
    evidence: list[Evidence] = []
    if successful is not None:
        data = successful.get("data")
        if isinstance(data, dict):
            provider_data = data.get("data")
            structured_data = (
                provider_data if isinstance(provider_data, dict) else data
            )
        evidence.append(
            Evidence(
                evidence_id=f"{spec.evidence_source_type}:{query}",
                task_id=spec.default_task_id,
                source_type=spec.evidence_source_type,
                provider=spec.evidence_provider,
                title=spec.evidence_title,
                content=f"{spec.evidence_title}查询：{query}",
            )
        )

    answer = ""
    for message in reversed(list(updates.get("messages") or [])):
        if isinstance(message, AIMessage):
            answer = str(message.content or "").strip()
            if answer:
                break
    if not answer:
        answer = spec.busy_answer

    if successful is not None:
        status = "completed"
        error_code = ""
        gaps: list[str] = []
    elif empty_success is not None:
        status = "failed"
        error_code = f"{required}_empty_result"
        gaps = ["问财未返回候选股票"]
    elif failed is not None:
        status = "failed"
        error_code = str(failed.get("error") or f"{required}_failed")
        gaps = ["必需选股工具调用失败"]
    else:
        status = "failed"
        error_code = f"{required}_not_called"
        gaps = ["必需选股工具未调用"]

    return AgentResult(
        task_id=spec.default_task_id,
        agent_id=spec.agent_id,
        status=status,
        answer=answer,
        structured_data=structured_data,
        evidence=evidence,
        gaps=gaps,
        error_code=error_code,
        metadata={
            "query": query,
            "skills": [document.name for document in active_binding.documents],
            "tool_ids": list(active_binding.tool_ids),
            "tool_results": tool_results,
        },
    )


__all__ = ["agent_result_from_tool_updates"]
