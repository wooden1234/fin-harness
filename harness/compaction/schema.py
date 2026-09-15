"""压缩摘要的结构化增量与合并后状态。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class CompactionFact(BaseModel):
    company: str = ""
    period: str = ""
    metric: str = ""
    value: str = ""
    unit: str = ""
    evidence_id: str = ""
    source_tool: str = ""
    note: str = ""
    turn: int = 0


class CompactionDelta(BaseModel):
    """facts 可来自窗口里任意更早轮次；open/completed 只允许来自当前 turn。"""

    new_facts: list[CompactionFact] = Field(default_factory=list)
    new_open_items: list[str] = Field(default_factory=list)
    new_completed_items: list[str] = Field(default_factory=list)
    narrative_delta: str = ""


class CompactionSummary(BaseModel):
    facts: list[CompactionFact] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    completed_items: list[str] = Field(default_factory=list)
    narrative: str = ""


Priority = Literal["low", "normal", "high", "critical"]
ItemStatus = Literal["active", "completed", "resolved", "superseded", "expired"]


class CompactionItemV2(BaseModel):
    """所有可合并上下文状态的公共生命周期字段。"""

    id: str = ""
    content: str = ""
    created_turn: int = 0
    updated_turn: int = 0
    status: ItemStatus = "active"
    priority: Priority = "normal"
    source: str = "compaction"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompactionFactV2(CompactionItemV2):
    company: str = ""
    period: str = ""
    metric: str = ""
    value: str = ""
    unit: str = ""
    evidence_id: str = ""
    source_tool: str = ""
    note: str = ""


_V2_COLLECTIONS = (
    "facts",
    "decisions",
    "constraints",
    "preferences",
    "entities",
    "skills",
    "resources",
    "tool_state",
    "completed_items",
    "open_items",
    "failures",
    "conversation_notes",
)


class CompactionDeltaV2(BaseModel):
    new_facts: list[CompactionFactV2] = Field(default_factory=list)
    new_decisions: list[CompactionItemV2] = Field(default_factory=list)
    new_constraints: list[CompactionItemV2] = Field(default_factory=list)
    new_preferences: list[CompactionItemV2] = Field(default_factory=list)
    new_entities: list[CompactionItemV2] = Field(default_factory=list)
    new_skills: list[CompactionItemV2] = Field(default_factory=list)
    new_resources: list[CompactionItemV2] = Field(default_factory=list)
    new_tool_state: list[CompactionItemV2] = Field(default_factory=list)
    new_completed_items: list[CompactionItemV2] = Field(default_factory=list)
    new_open_items: list[CompactionItemV2] = Field(default_factory=list)
    new_failures: list[CompactionItemV2] = Field(default_factory=list)
    new_conversation_notes: list[CompactionItemV2] = Field(default_factory=list)
    resolved_ids: list[str] = Field(default_factory=list)
    superseded_ids: list[str] = Field(default_factory=list)
    narrative_delta: str = ""


class CompactionSummaryV2(BaseModel):
    schema_version: int = 2
    facts: list[CompactionFactV2] = Field(default_factory=list)
    decisions: list[CompactionItemV2] = Field(default_factory=list)
    constraints: list[CompactionItemV2] = Field(default_factory=list)
    preferences: list[CompactionItemV2] = Field(default_factory=list)
    entities: list[CompactionItemV2] = Field(default_factory=list)
    skills: list[CompactionItemV2] = Field(default_factory=list)
    resources: list[CompactionItemV2] = Field(default_factory=list)
    tool_state: list[CompactionItemV2] = Field(default_factory=list)
    completed_items: list[CompactionItemV2] = Field(default_factory=list)
    open_items: list[CompactionItemV2] = Field(default_factory=list)
    failures: list[CompactionItemV2] = Field(default_factory=list)
    conversation_notes: list[CompactionItemV2] = Field(default_factory=list)
    narrative: str = ""


def stable_item_id(kind: str, payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{kind}-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _legacy_item(kind: str, content: str, *, turn: int = 0) -> CompactionItemV2:
    return CompactionItemV2(
        id=stable_item_id(kind, {"content": content, "turn": turn}),
        content=content,
        created_turn=turn,
        updated_turn=turn,
        source="legacy",
    )


def summary_v1_to_v2(summary: CompactionSummary) -> CompactionSummaryV2:
    facts: list[CompactionFactV2] = []
    for fact in summary.facts:
        raw = fact.model_dump()
        turn = int(raw.pop("turn", 0) or 0)
        facts.append(
            CompactionFactV2(
                **raw,
                id=stable_item_id(
                    "fact",
                    {key: raw.get(key) for key in ("company", "period", "metric", "note", "evidence_id")},
                ),
                content=fact.note,
                created_turn=turn,
                updated_turn=turn,
                source=fact.source_tool or "legacy",
            )
        )
    return CompactionSummaryV2(
        facts=facts,
        open_items=[_legacy_item("open", value) for value in summary.open_items],
        completed_items=[_legacy_item("completed", value) for value in summary.completed_items],
        narrative=summary.narrative,
    )


def render_summary_text_v2(summary: CompactionSummaryV2) -> str:
    lines = ["[压缩摘要 v2]"]
    labels = {
        "facts": "已确认事实",
        "decisions": "已定决策",
        "constraints": "有效约束",
        "preferences": "用户偏好",
        "entities": "实体与消歧",
        "skills": "技能状态",
        "resources": "资源",
        "tool_state": "工具状态",
        "completed_items": "已完成",
        "open_items": "未完成",
        "failures": "未解决失败",
        "conversation_notes": "对话说明",
    }
    for field in _V2_COLLECTIONS:
        values = [item for item in getattr(summary, field) if item.status not in {"resolved", "superseded", "expired"}]
        rendered: list[str] = []
        for item in values:
            if isinstance(item, CompactionFactV2):
                legacy = CompactionFact(
                    company=item.company,
                    period=item.period,
                    metric=item.metric,
                    value=item.value,
                    unit=item.unit,
                    evidence_id=item.evidence_id,
                    source_tool=item.source_tool,
                    note=item.note or item.content,
                    turn=item.updated_turn,
                )
                text = _fact_line(legacy)
            else:
                text = item.content
            if text:
                rendered.append(text)
        if rendered:
            lines.append(f"{labels[field]}：")
            lines.extend(f"- {value}" for value in rendered)
    if summary.narrative.strip():
        lines.append(f"补充：{summary.narrative.strip()}")
    return "\n".join(lines).strip()


def _fact_line(fact: CompactionFact) -> str:
    head = " ".join(part for part in (fact.company, fact.period, fact.metric) if part).strip()
    value = f"{fact.value}{fact.unit}".strip()
    if head and value:
        body = f"{head}={value}"
    else:
        body = head or value or fact.note.strip()
    extras: list[str] = []
    if fact.evidence_id:
        extras.append(f"evidence_id={fact.evidence_id}")
    if fact.source_tool:
        extras.append(f"来源={fact.source_tool}")
    if fact.note and fact.note.strip() not in body:
        extras.append(fact.note.strip())
    if extras:
        body = f"{body}（{'，'.join(extras)}）"
    return body


def render_summary_text(summary: CompactionSummary) -> str:
    lines = ["[压缩摘要]"]
    facts = [_fact_line(item) for item in summary.facts if _fact_line(item)]
    if facts:
        lines.append("已确认事实：")
        lines.extend(f"- {item}" for item in facts)
    if summary.completed_items:
        lines.append("本轮已完成：")
        lines.extend(f"- {item}" for item in summary.completed_items)
    if summary.open_items:
        lines.append("本轮未完成：")
        lines.extend(f"- {item}" for item in summary.open_items)
    narrative = str(summary.narrative or "").strip()
    if narrative:
        lines.append(f"说明：{narrative}")
    return "\n".join(lines).strip()


__all__ = [
    "CompactionDelta",
    "CompactionFact",
    "CompactionSummary",
    "render_summary_text",
    "CompactionDeltaV2",
    "CompactionFactV2",
    "CompactionItemV2",
    "CompactionSummaryV2",
    "render_summary_text_v2",
    "stable_item_id",
    "summary_v1_to_v2",
]
