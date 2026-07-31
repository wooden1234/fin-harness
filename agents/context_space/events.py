"""上下文治理事件的脱敏写入适配层。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from agents.context_space.models import (
    ContextCounters,
    ContextMeasurement,
    ContextSpaceType,
)
from agents.runtime_context import AgentRuntimeContext


async def record_context_event(
    runtime: Runtime[AgentRuntimeContext] | None,
    *,
    space_type: ContextSpaceType,
    event_type: str,
    measurement: ContextMeasurement | None = None,
    counters: ContextCounters | None = None,
    details: Mapping[str, Any] | None = None,
    agent_id: str | None = None,
    parent_space_id: str | None = None,
) -> None:
    """仅生产运行写库；Studio 和无 run_id 测试不会产生副作用。"""
    context = runtime.context if runtime is not None else None
    run_id = str(getattr(context, "run_id", "") or "")
    user_id = str(getattr(context, "user_id", "") or "")
    if not run_id or not user_id.isdigit() or int(user_id) <= 0:
        return

    from app.services.agent.context_event_service import ContextEventService

    active_counters = counters or ContextCounters()
    conversation_id_text = str(getattr(context, "conversation_id", "") or "")
    conversation_id = (
        int(conversation_id_text) if conversation_id_text.isdigit() else None
    )
    space_id = f"{space_type.value}:{run_id}"
    await ContextEventService.record(
        tenant_id=str(getattr(context, "tenant_id", "default")),
        user_id=int(user_id),
        conversation_id=conversation_id,
        run_id=run_id,
        agent_id=agent_id,
        space_type=space_type.value,
        space_id=space_id,
        parent_space_id=parent_space_id,
        event_type=event_type,
        estimated_tokens=(measurement.estimated_tokens if measurement else None),
        actual_input_tokens=(measurement.actual_input_tokens if measurement else None),
        effective_limit=(measurement.effective_limit if measurement else None),
        compaction_round_count=active_counters.compaction_round_count,
        summary_attempt_count=active_counters.summary_attempt_count,
        snip_count=active_counters.snip_count,
        provider_retry_count=active_counters.provider_retry_count,
        details={
            "counter_kind": measurement.counter_kind if measurement else None,
            "utilization_ratio": (
                measurement.utilization_ratio if measurement else None
            ),
            "trigger_exceeded": (
                measurement.trigger_exceeded if measurement else None
            ),
            **dict(details or {}),
        },
    )


__all__ = ["record_context_event"]
