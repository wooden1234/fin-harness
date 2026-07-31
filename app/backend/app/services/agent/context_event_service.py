"""上下文空间事件写入、查询与聚合。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.core.logger import get_logger
from app.models.agent.agent_run import AgentRun
from app.models.agent.agent_run_event import AgentRunEvent

logger = get_logger(service="context_event")

_ALLOWED_DETAIL_KEYS = frozenset(
    {
        "counter_kind",
        "utilization_ratio",
        "estimate_error_ratio",
        "message_count_before",
        "message_count_after",
        "evidence_count_before",
        "evidence_count_after",
        "evidence_ids",
        "content_hash",
        "block_type",
        "error_code",
        "duration_ms",
        "admitted",
        "trigger_exceeded",
    }
)


def _safe_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """事件禁止携带原始消息、工具结果、摘要正文或隐藏推理。"""
    if not details:
        return {}
    safe: dict[str, Any] = {}
    for key, value in details.items():
        if key not in _ALLOWED_DETAIL_KEYS:
            continue
        if key == "evidence_ids":
            safe[key] = [str(item)[:128] for item in list(value or [])[:64]]
        elif isinstance(value, str):
            safe[key] = value[:256]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key] = value
    return safe


class ContextEventService:
    @staticmethod
    async def record(
        *,
        tenant_id: str,
        user_id: int,
        run_id: str,
        space_type: str,
        space_id: str,
        event_type: str,
        conversation_id: int | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        parent_space_id: str | None = None,
        event_key: str | None = None,
        estimated_tokens: int | None = None,
        actual_input_tokens: int | None = None,
        effective_limit: int | None = None,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        compaction_round_count: int = 0,
        summary_attempt_count: int = 0,
        snip_count: int = 0,
        provider_retry_count: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """最佳努力写事件；监控故障不阻断 Agent。"""
        try:
            async with AsyncSessionLocal() as db:
                db.add(
                    AgentRunEvent(
                        event_key=event_key or f"context:{run_id}:{uuid4()}",
                        tenant_id=str(tenant_id),
                        user_id=int(user_id),
                        conversation_id=conversation_id,
                        run_id=str(run_id),
                        task_id=task_id,
                        agent_id=agent_id,
                        space_type=space_type,
                        space_id=space_id,
                        parent_space_id=parent_space_id,
                        event_type=event_type,
                        estimated_tokens=estimated_tokens,
                        actual_input_tokens=actual_input_tokens,
                        effective_limit=effective_limit,
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        compaction_round_count=compaction_round_count,
                        summary_attempt_count=summary_attempt_count,
                        snip_count=snip_count,
                        provider_retry_count=provider_retry_count,
                        details=_safe_details(details),
                    )
                )
                await db.commit()
        except Exception as exc:
            logger.warning("context event write failed: {}", type(exc).__name__)

    @staticmethod
    async def _owned_run(
        *,
        run_id: str,
        tenant_id: str,
        user_id: int,
        platform_admin: bool,
    ) -> AgentRun:
        async with AsyncSessionLocal() as db:
            run = await db.get(AgentRun, run_id)
            if run is None:
                raise ValueError("agent_run_not_found")
            if not platform_admin and (
                str(run.tenant_id) != str(tenant_id) or int(run.user_id) != int(user_id)
            ):
                raise PermissionError("agent_run_forbidden")
            return run

    @classmethod
    async def list_events(
        cls,
        *,
        run_id: str,
        tenant_id: str,
        user_id: int,
        platform_admin: bool = False,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[AgentRunEvent]:
        await cls._owned_run(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            platform_admin=platform_admin,
        )
        async with AsyncSessionLocal() as db:
            rows = await db.execute(
                select(AgentRunEvent)
                .where(
                    AgentRunEvent.run_id == run_id,
                    AgentRunEvent.id > max(0, int(after_id)),
                )
                .order_by(AgentRunEvent.id.asc())
                .limit(max(1, min(int(limit), 500)))
            )
            return list(rows.scalars().all())

    @classmethod
    async def metrics(cls, **kwargs: Any) -> dict[str, Any]:
        events: list[AgentRunEvent] = []
        after_id = 0
        while True:
            page = await cls.list_events(
                limit=500,
                after_id=after_id,
                **kwargs,
            )
            events.extend(page)
            if len(page) < 500:
                break
            after_id = int(page[-1].id)
        by_space: dict[str, dict[str, Any]] = {}
        totals = {
            "context_provider_overflow_total": 0,
            "context_post_compaction_overflow_total": 0,
            "context_snip_total": 0,
            "context_admission_rejected_total": 0,
            "context_max_rounds_exhausted_total": 0,
            "context_estimate_underflow_total": 0,
        }
        for event in events:
            item = by_space.setdefault(
                event.space_id,
                {
                    "space_type": event.space_type,
                    "max_estimated_tokens": 0,
                    "max_actual_input_tokens": 0,
                    "compaction_round_count": 0,
                    "summary_attempt_count": 0,
                    "snip_count": 0,
                    "provider_retry_count": 0,
                    "provider_overflow_count": 0,
                    "admission_rejected_count": 0,
                },
            )
            item["max_estimated_tokens"] = max(
                item["max_estimated_tokens"], event.estimated_tokens or 0
            )
            item["max_actual_input_tokens"] = max(
                item["max_actual_input_tokens"], event.actual_input_tokens or 0
            )
            for field in (
                "compaction_round_count",
                "summary_attempt_count",
                "snip_count",
                "provider_retry_count",
            ):
                item[field] = max(item[field], int(getattr(event, field) or 0))
            if event.event_type == "context.provider_overflow":
                item["provider_overflow_count"] += 1
                totals["context_provider_overflow_total"] += 1
                if int(event.compaction_round_count or 0) > 0:
                    totals["context_post_compaction_overflow_total"] += 1
            if event.event_type == "context.admission_rejected":
                item["admission_rejected_count"] += 1
                totals["context_admission_rejected_total"] += 1
            if event.event_type == "context.snip_applied":
                totals["context_snip_total"] += 1
            if (
                event.event_type == "context.degraded"
                and (event.details or {}).get("error_code")
                == "max_compaction_rounds_exhausted"
            ):
                totals["context_max_rounds_exhausted_total"] += 1
            estimated = event.estimated_tokens
            actual = event.actual_input_tokens
            if (
                estimated is not None
                and actual is not None
                and actual > estimated
            ):
                totals["context_estimate_underflow_total"] += 1
        return {
            "run_id": kwargs["run_id"],
            "spaces": by_space,
            **totals,
        }


__all__ = ["ContextEventService"]
