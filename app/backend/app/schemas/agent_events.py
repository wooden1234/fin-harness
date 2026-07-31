"""Agent 运行上下文事件查询响应。"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentRunEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    task_id: str | None
    agent_id: str | None
    space_type: str
    space_id: str
    parent_space_id: str | None
    event_type: str
    estimated_tokens: int | None
    actual_input_tokens: int | None
    effective_limit: int | None
    tokens_before: int | None
    tokens_after: int | None
    compaction_round_count: int
    summary_attempt_count: int
    snip_count: int
    provider_retry_count: int
    details: dict[str, Any]
    created_at: datetime


class AgentRunEventPage(BaseModel):
    items: list[AgentRunEventRead]
    next_after_id: int | None


__all__ = ["AgentRunEventPage", "AgentRunEventRead"]
