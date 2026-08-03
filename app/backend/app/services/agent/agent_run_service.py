"""Agent 运行记录服务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, or_, select

from app.core.database import AsyncSessionLocal
from app.models.agent.agent_run import AgentRun, AgentRunStatus


def summarize_run_snapshots(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """从低基数字段生成可用于灰度闸门的运行指标。"""
    valid = [item for item in snapshots if isinstance(item, dict)]
    durations = [
        float(value)
        for item in valid
        if isinstance((value := item.get("duration_ms")), (int, float))
        and not isinstance(value, bool)
    ]
    routes = _count_values(valid, "route")
    statuses = _count_values(valid, "execution_status")
    by_budget: dict[str, dict[str, float | int]] = {}
    for tier in sorted(
        {str(item.get("budget_tier")) for item in valid if item.get("budget_tier")}
    ):
        tier_durations = [
            float(item["duration_ms"])
            for item in valid
            if item.get("budget_tier") == tier
            and isinstance(item.get("duration_ms"), (int, float))
            and not isinstance(item.get("duration_ms"), bool)
        ]
        by_budget[tier] = _duration_summary(tier_durations)

    cited = sum(bool(item.get("citations")) for item in valid)
    answered = sum(bool(str(item.get("content") or "").strip()) for item in valid)
    grounded = [
        item
        for item in valid
        if item.get("execution_mode") in {"tool_assisted", "deep_research", "partial"}
    ]
    salvaged = sum(bool(item.get("salvaged")) for item in valid)
    salvage_eligible = sum(bool(item.get("salvage_eligible")) for item in valid)
    timed_out = sum(bool(item.get("timed_out")) for item in valid)
    unauthorized_attempts = sum(
        int(item.get("unauthorized_tool_attempts") or 0) for item in valid
    )
    sample_size = len(valid)

    return {
        "sample_size": sample_size,
        "duration_ms": _duration_summary(durations),
        "by_budget": by_budget,
        "routes": routes,
        "execution_statuses": statuses,
        "citation_coverage_ratio": round(cited / sample_size, 4)
        if sample_size
        else 0.0,
        "answer_availability_ratio": round(answered / sample_size, 4)
        if sample_size
        else 0.0,
        "model_rounds": _duration_summary(
            [float(item.get("model_rounds") or 0) for item in valid]
        ),
        "tool_calls": _duration_summary(
            [float(item.get("tool_calls") or 0) for item in valid]
        ),
        "source_family_count": _duration_summary(
            [float(len(item.get("source_families") or [])) for item in valid]
        ),
        "evidence_count": _duration_summary(
            [float(item.get("evidence_count") or 0) for item in valid]
        ),
        "statement_coverage_ratio": round(
            sum(float(item.get("statement_coverage") or 0) for item in grounded)
            / len(grounded),
            4,
        ) if grounded else 1.0,
        "salvage_count": salvaged,
        "salvage_ratio": round(salvaged / salvage_eligible, 4)
        if salvage_eligible
        else 1.0,
        "salvage_eligible_count": salvage_eligible,
        "timeout_ratio": round(timed_out / sample_size, 4) if sample_size else 0.0,
        "unauthorized_tool_attempts": unauthorized_attempts,
        "cross_period_mismatch": sum(int(item.get("cross_period_mismatch") or 0) for item in valid),
        "tool_budget_exhausted": sum(int(item.get("tool_budget_exhausted") or 0) for item in valid),
        "insufficient_tool_result": sum(int(item.get("insufficient_tool_result") or 0) for item in valid),
        "raw_json_leak": sum(bool(item.get("raw_json_leak")) for item in valid),
        "conflict_detected": sum(int(item.get("conflict_detected") or 0) for item in valid),
        "conflict_resolved": sum(int(item.get("conflict_resolved") or 0) for item in valid),
        "completed_with_gaps": sum(bool(item.get("completed_with_gaps")) for item in valid),
    }


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return round(ordered[index], 2)


def _count_values(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


class AgentRunService:
    """集中管理 AgentRun 的状态流转。"""

    @staticmethod
    async def create_run(
        *,
        user_id: int,
        conversation_id: int | None,
        thread_id: str,
        tenant_id: str = "default",
        trace_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentRun:
        async with AsyncSessionLocal() as db:
            run = AgentRun(
                id=run_id,
                user_id=user_id,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                trace_id=trace_id,
                status=AgentRunStatus.ACCEPTED.value,
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)
            return run

    @staticmethod
    async def _update(run_id: str, **values: Any) -> AgentRun:
        async with AsyncSessionLocal() as db:
            run = await db.get(AgentRun, run_id)
            if run is None:
                raise ValueError(f"agent run 不存在: {run_id}")
            for key, value in values.items():
                setattr(run, key, value)
            run.updated_at = func.now()
            await db.commit()
            await db.refresh(run)
            return run

    @classmethod
    async def mark_running(cls, run_id: str) -> AgentRun:
        return await cls._update(
            run_id,
            status=AgentRunStatus.RUNNING.value,
            started_at=datetime.now(timezone.utc),
        )

    @classmethod
    async def mark_graph_completed(
        cls,
        run_id: str,
        *,
        checkpoint_id: str | None = None,
        summary_snapshot: dict[str, Any] | None = None,
    ) -> AgentRun:
        return await cls._update(
            run_id,
            status=AgentRunStatus.GRAPH_COMPLETED.value,
            checkpoint_id=checkpoint_id,
            summary_snapshot=summary_snapshot,
        )

    @classmethod
    async def reconcile_stale_runs(cls, *, timeout_seconds: int = 900) -> int:
        """将超时未更新的运行标记为失败，避免永久停留在 running。"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(AgentRun).where(
                    AgentRun.status.in_(
                        [AgentRunStatus.ACCEPTED.value, AgentRunStatus.RUNNING.value]
                    ),
                    or_(AgentRun.updated_at < cutoff, AgentRun.updated_at.is_(None)),
                )
            )
            runs = result.scalars().all()
            for run in runs:
                run.status = AgentRunStatus.FAILED.value
                run.error_code = "stale_run_timeout"
                run.error_message = f"运行超过 {timeout_seconds} 秒未更新"
                run.completed_at = datetime.now(timezone.utc)
                run.updated_at = func.now()
            await db.commit()
            return len(runs)

    @staticmethod
    async def metrics() -> dict[str, Any]:
        from app.models.persistence.outbox_event import OutboxEvent

        async with AsyncSessionLocal() as db:
            run_rows = (await db.execute(
                select(AgentRun.status, func.count(AgentRun.id)).group_by(AgentRun.status)
            )).all()
            outbox_rows = (await db.execute(
                select(OutboxEvent.status, func.count(OutboxEvent.id)).group_by(OutboxEvent.status)
            )).all()
            snapshot_rows = (
                await db.execute(
                    select(AgentRun.summary_snapshot)
                    .where(AgentRun.summary_snapshot.is_not(None))
                    .order_by(AgentRun.created_at.desc())
                    .limit(1000)
                )
            ).scalars().all()
            result = {f"agent_runs_{status}": int(count) for status, count in run_rows}
            result.update({f"outbox_{status}": int(count) for status, count in outbox_rows})
            result["runs"] = summarize_run_snapshots(list(snapshot_rows))
            return result

    @classmethod
    async def mark_persisted(
        cls, run_id: str, *, response_message_id: int | None = None
    ) -> AgentRun:
        # response_message_id 预留给后续 message.run_id 关联，不影响当前表结构。
        _ = response_message_id
        return await cls._update(
            run_id,
            status=AgentRunStatus.PERSISTED.value,
            completed_at=datetime.now(timezone.utc),
            error_code=None,
            error_message=None,
        )

    @classmethod
    async def mark_persist_pending(
        cls, run_id: str, *, error_code: str = "message_persist_failed", error_message: str = ""
    ) -> AgentRun:
        return await cls._update(
            run_id,
            status=AgentRunStatus.PERSIST_PENDING.value,
            completed_at=datetime.now(timezone.utc),
            error_code=error_code,
            error_message=error_message[:4000],
        )

    @classmethod
    async def mark_failed(
        cls, run_id: str, *, error_code: str = "agent_run_failed", error_message: str = ""
    ) -> AgentRun:
        return await cls._update(
            run_id,
            status=AgentRunStatus.FAILED.value,
            completed_at=datetime.now(timezone.utc),
            error_code=error_code,
            error_message=error_message[:4000],
        )
