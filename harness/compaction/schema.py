"""压缩摘要的结构化增量与合并后状态。"""

from __future__ import annotations

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
]
