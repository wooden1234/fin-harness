"""结构化会话摘要的确定性合并、裁剪和渲染。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any, Literal
from uuid import uuid4

from agents.context_compressor.models import (
    ConversationSummaryPatch,
    ConversationSummaryV2,
    EntityRef,
    FinanceContext,
    TopicConstraint,
    TopicDelta,
    TopicSummary,
)

ProjectionPurpose = Literal["rewrite", "routing", "planning", "answer"]
_CONTROL_PATTERN = re.compile(
    r"(忽略|绕过|覆盖).{0,12}(系统|开发者|安全|规则)|"
    r"(system|developer)\s*(prompt|message)|"
    r"(扮演|切换).{0,8}(角色|身份)",
    re.IGNORECASE,
)


def parse_summary_v2(value: Any) -> ConversationSummaryV2 | None:
    if isinstance(value, ConversationSummaryV2):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return ConversationSummaryV2.model_validate(value)
    except Exception:
        return None


def _safe_text(value: str, *, limit: int = 300) -> str | None:
    normalized = " ".join(str(value).split()).strip()
    if not normalized or _CONTROL_PATTERN.search(normalized):
        return None
    return normalized[:limit]


def _unique_text(values: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _safe_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _unique_entities(values: list[EntityRef], *, limit: int = 16) -> list[EntityRef]:
    result: list[EntityRef] = []
    seen: set[tuple[str, str, str | None]] = set()
    for entity in values:
        name = _safe_text(entity.name, limit=80)
        if not name:
            continue
        cleaned = entity.model_copy(update={"name": name})
        key = (cleaned.name, cleaned.entity_type, cleaned.identifier)
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _apply_delta(topic: TopicSummary, delta: TopicDelta, revision: int) -> TopicSummary:
    remove_entities = set(delta.remove_entity_names)
    entities = [item for item in topic.entities if item.name not in remove_entities]
    entities = _unique_entities([*entities, *delta.add_entities])

    facts = [item for item in topic.facts if item not in set(delta.remove_facts)]
    decisions = [item for item in topic.decisions if item not in set(delta.remove_decisions)]
    questions = [
        item for item in topic.open_questions
        if item not in set(delta.resolve_open_questions)
    ]

    constraints = {
        item.name: item
        for item in topic.constraints
        if item.name not in set(delta.remove_constraint_names)
    }
    for item in delta.upsert_constraints:
        if _safe_text(item.name, limit=64) and _safe_text(item.value, limit=200):
            constraints[item.name] = item

    finance = topic.finance_context or FinanceContext()
    finance = finance.model_copy(
        update={
            "securities": _unique_text(
                [
                    *[item for item in finance.securities if item not in set(delta.remove_securities)],
                    *delta.add_securities,
                ],
                limit=12,
            ),
            "time_ranges": (
                _unique_text(delta.set_time_ranges, limit=8)
                if delta.set_time_ranges is not None
                else finance.time_ranges
            ),
            "metrics": _unique_text(
                [
                    *[item for item in finance.metrics if item not in set(delta.remove_metrics)],
                    *delta.add_metrics,
                ],
                limit=16,
            ),
            "currency_unit": (
                _safe_text(delta.set_currency_unit, limit=32)
                if delta.set_currency_unit is not None
                else finance.currency_unit
            ),
        }
    )
    has_finance = bool(
        finance.securities
        or finance.time_ranges
        or finance.metrics
        or finance.currency_unit
    )
    return topic.model_copy(
        update={
            "title": _safe_text(delta.title, limit=100) if delta.title else topic.title,
            "status": delta.status or topic.status,
            "entities": entities,
            "facts": _unique_text([*facts, *delta.add_facts], limit=16),
            "constraints": list(constraints.values())[:12],
            "decisions": _unique_text([*decisions, *delta.add_decisions], limit=12),
            "open_questions": _unique_text([*questions, *delta.add_open_questions], limit=12),
            "finance_context": finance if has_finance else None,
            "last_touched_revision": revision,
        }
    )


def _new_topic(payload: Any, revision: int) -> TopicSummary:
    topic_id = f"topic-{uuid4()}"
    return TopicSummary(
        topic_id=topic_id,
        title=_safe_text(payload.title, limit=100) or "未命名话题",
        domain=payload.domain,
        status="paused",
        entities=_unique_entities(payload.entities),
        facts=_unique_text(payload.facts, limit=16),
        constraints=payload.constraints[:12],
        decisions=_unique_text(payload.decisions, limit=12),
        open_questions=_unique_text(payload.open_questions, limit=12),
        finance_context=payload.finance_context,
        last_touched_revision=revision,
    )


def apply_summary_patch(
    current: ConversationSummaryV2 | None,
    patch: ConversationSummaryPatch,
) -> ConversationSummaryV2:
    """应用 Patch；未知引用立即失败，避免静默写入错误话题。"""
    base = deepcopy(current) if current is not None else ConversationSummaryV2()
    revision = base.revision + 1
    topics = {item.topic_id: item for item in base.topics}
    temporary_refs: dict[str, str] = {}

    for payload in patch.new_topics:
        if payload.temporary_ref in temporary_refs or payload.temporary_ref in topics:
            raise ValueError("duplicate_new_topic_reference")
        topic = _new_topic(payload, revision)
        temporary_refs[payload.temporary_ref] = topic.topic_id
        topics[topic.topic_id] = topic

    for delta in patch.topic_deltas:
        if delta.topic_id not in topics:
            raise ValueError(f"unknown_topic_id:{delta.topic_id}")
        topics[delta.topic_id] = _apply_delta(topics[delta.topic_id], delta, revision)

    active_ref = patch.activate_topic_ref
    active_id = temporary_refs.get(active_ref, active_ref) if active_ref else base.active_topic_id
    if active_id is not None and active_id not in topics:
        raise ValueError(f"unknown_active_topic_ref:{active_ref}")
    if active_id is None and topics:
        active_id = max(topics.values(), key=lambda item: item.last_touched_revision).topic_id

    normalized: list[TopicSummary] = []
    for topic in topics.values():
        status = "active" if topic.topic_id == active_id else (
            "resolved" if topic.status == "resolved" else "paused"
        )
        normalized.append(topic.model_copy(update={"status": status}))

    active = [item for item in normalized if item.topic_id == active_id]
    paused = sorted(
        [item for item in normalized if item.topic_id != active_id and item.status != "resolved"],
        key=lambda item: item.last_touched_revision,
        reverse=True,
    )[:2]
    kept = [*active, *paused]
    return ConversationSummaryV2(
        revision=revision,
        active_topic_id=active_id if active else None,
        topics=kept,
    )


def _topic_lines(topic: TopicSummary, *, detailed: bool) -> list[str]:
    lines = [f"话题：{topic.title}（{topic.domain}，{topic.status}）"]
    if topic.entities:
        lines.append("实体：" + "、".join(item.name for item in topic.entities))
    if detailed and topic.facts:
        lines.append("事实：" + "；".join(topic.facts))
    if topic.constraints:
        lines.append(
            "约束：" + "；".join(f"{item.name}={item.value}" for item in topic.constraints)
        )
    if topic.decisions:
        lines.append("决策：" + "；".join(topic.decisions))
    if topic.open_questions:
        lines.append("未决问题：" + "；".join(topic.open_questions))
    if detailed and topic.finance_context:
        finance = topic.finance_context
        if finance.securities:
            lines.append("证券：" + "、".join(finance.securities))
        if finance.time_ranges:
            lines.append("时间：" + "、".join(finance.time_ranges))
        if finance.metrics:
            lines.append("指标：" + "、".join(finance.metrics))
        if finance.currency_unit:
            lines.append("单位：" + finance.currency_unit)
    return lines


def render_summary_v2(
    summary: ConversationSummaryV2,
    *,
    purpose: ProjectionPurpose = "answer",
) -> str:
    """按消费者目的投影，避免所有节点读取整个摘要。"""
    active = next(
        (item for item in summary.topics if item.topic_id == summary.active_topic_id),
        None,
    )
    if active is None:
        return ""
    parts = ["\n".join(_topic_lines(active, detailed=True))]
    if purpose == "rewrite":
        paused = [item for item in summary.topics if item.topic_id != active.topic_id]
        for topic in paused:
            names = "、".join(item.name for item in topic.entities)
            parts.append(f"暂停话题：{topic.title}" + (f"；实体：{names}" if names else ""))
    elif purpose == "answer":
        for topic in summary.topics:
            if topic.topic_id != active.topic_id and topic.decisions:
                parts.append(f"此前话题 {topic.title} 的决策：" + "；".join(topic.decisions))
    return "\n\n".join(parts)


__all__ = [
    "ProjectionPurpose",
    "apply_summary_patch",
    "parse_summary_v2",
    "render_summary_v2",
]
