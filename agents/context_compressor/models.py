"""结构化会话摘要及增量 Patch 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


TopicStatus = Literal["active", "paused", "resolved"]
TopicDomain = Literal["general", "finance", "travel", "other"]


class EntityRef(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    entity_type: str = Field(default="unknown", max_length=32)
    identifier: str | None = Field(default=None, max_length=64)


class TopicConstraint(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=200)


class FinanceContext(BaseModel):
    securities: list[str] = Field(default_factory=list, max_length=12)
    time_ranges: list[str] = Field(default_factory=list, max_length=8)
    metrics: list[str] = Field(default_factory=list, max_length=16)
    currency_unit: str | None = Field(default=None, max_length=32)


class TopicPayload(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    domain: TopicDomain = "general"
    entities: list[EntityRef] = Field(default_factory=list, max_length=16)
    facts: list[str] = Field(default_factory=list, max_length=16)
    constraints: list[TopicConstraint] = Field(default_factory=list, max_length=12)
    decisions: list[str] = Field(default_factory=list, max_length=12)
    open_questions: list[str] = Field(default_factory=list, max_length=12)
    finance_context: FinanceContext | None = None


class TopicSummary(TopicPayload):
    topic_id: str = Field(min_length=1, max_length=80)
    status: TopicStatus = "paused"
    last_touched_revision: int = Field(default=0, ge=0)


class ConversationSummaryV2(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    revision: int = Field(default=0, ge=0)
    active_topic_id: str | None = None
    topics: list[TopicSummary] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_active_topic(self) -> "ConversationSummaryV2":
        ids = [topic.topic_id for topic in self.topics]
        if len(ids) != len(set(ids)):
            raise ValueError("conversation_topic_ids_must_be_unique")
        if self.active_topic_id is not None and self.active_topic_id not in ids:
            raise ValueError("active_topic_id_must_exist")
        return self


class NewTopic(TopicPayload):
    temporary_ref: str = Field(min_length=1, max_length=64)


class TopicDelta(BaseModel):
    topic_id: str = Field(min_length=1, max_length=80)
    title: str | None = Field(default=None, min_length=1, max_length=100)
    status: TopicStatus | None = None
    add_entities: list[EntityRef] = Field(default_factory=list, max_length=16)
    remove_entity_names: list[str] = Field(default_factory=list, max_length=16)
    add_facts: list[str] = Field(default_factory=list, max_length=16)
    remove_facts: list[str] = Field(default_factory=list, max_length=16)
    upsert_constraints: list[TopicConstraint] = Field(default_factory=list, max_length=12)
    remove_constraint_names: list[str] = Field(default_factory=list, max_length=12)
    add_decisions: list[str] = Field(default_factory=list, max_length=12)
    remove_decisions: list[str] = Field(default_factory=list, max_length=12)
    add_open_questions: list[str] = Field(default_factory=list, max_length=12)
    resolve_open_questions: list[str] = Field(default_factory=list, max_length=12)
    add_securities: list[str] = Field(default_factory=list, max_length=12)
    remove_securities: list[str] = Field(default_factory=list, max_length=12)
    set_time_ranges: list[str] | None = Field(default=None, max_length=8)
    add_metrics: list[str] = Field(default_factory=list, max_length=16)
    remove_metrics: list[str] = Field(default_factory=list, max_length=16)
    set_currency_unit: str | None = Field(default=None, max_length=32)


class ConversationSummaryPatch(BaseModel):
    """LLM 只提出语义变更，本地代码负责 ID、版本和容量。"""

    new_topics: list[NewTopic] = Field(default_factory=list, max_length=3)
    topic_deltas: list[TopicDelta] = Field(default_factory=list, max_length=6)
    activate_topic_ref: str | None = Field(default=None, max_length=80)


__all__ = [
    "ConversationSummaryPatch",
    "ConversationSummaryV2",
    "EntityRef",
    "FinanceContext",
    "NewTopic",
    "TopicConstraint",
    "TopicDelta",
    "TopicSummary",
]
