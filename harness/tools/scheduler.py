"""工具调度：可并行的一起跑，结果按模型顺序提交。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from harness.llm.types import ToolCallDraft
from harness.session.types import EventDraft, SessionEvent, new_id
from harness.tools.arguments import coerce_tool_arguments
from harness.tools.definition import ToolDefinition
from harness.tools.errors import enrich_tool_result, error_result
from harness.tools.pipeline import ToolPipeline
from harness.tools.retry_policy import (
    allocate_tool_attempts,
    retry_exhausted_result,
    tool_attempt_counts,
    unknown_tool_result,
)


class ToolResolver(Protocol):
    def resolve(self, name: str) -> ToolDefinition | None: ...

    pipeline: ToolPipeline


@dataclass
class SchedulerOutcome:
    waiting_approval: bool = False
    approval_id: str | None = None
    call_id: str | None = None
    name: str | None = None
    results: list[dict[str, Any]] | None = None


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    payloads = coerce_tool_arguments(raw)
    if not payloads:
        text = str(raw or "").strip()
        return {} if not text else None
    return payloads[0]


async def execute_tool_calls(
    *,
    store: Any,
    session_id: str,
    runtime: ToolResolver,
    calls: Sequence[ToolCallDraft],
    turn: int,
    step: int,
    run_id: str,
    abort: asyncio.Event,
) -> SchedulerOutcome:
    if not calls:
        return SchedulerOutcome(results=[])
    approval_call = None
    for call in calls:
        definition = runtime.resolve(call.name)
        if definition is not None and definition.requires_human_approval:
            approval_call = call
            break
    if approval_call is not None:
        approval_id = new_id()
        for call in calls:
            await store.append(
                session_id,
                EventDraft(
                    event_type="tool/call",
                    turn=turn,
                    step=step,
                    run_id=run_id,
                    data={
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                ),
            )
            if call.call_id == approval_call.call_id:
                continue
            deferred = error_result("deferred_for_approval", name=call.name)
            await store.append(
                session_id,
                EventDraft(
                    event_type="tool/result",
                    turn=turn,
                    step=step,
                    run_id=run_id,
                    surface_op="append",
                    data={
                        "call_id": call.call_id,
                        "name": call.name,
                        "ok": False,
                        "content": json.dumps(deferred, ensure_ascii=False),
                    },
                ),
            )
        await store.append(
            session_id,
            EventDraft(
                event_type="approval/asked",
                turn=turn,
                step=step,
                run_id=run_id,
                correlation_id=approval_id,
                data={
                    "approval_id": approval_id,
                    "call_id": approval_call.call_id,
                    "name": approval_call.name,
                    "arguments": approval_call.arguments,
                },
            ),
        )
        return SchedulerOutcome(
            waiting_approval=True,
            approval_id=approval_id,
            call_id=approval_call.call_id,
            name=approval_call.name,
        )

    events = await store.load_events(session_id)
    blocked_ids, _counts = allocate_tool_attempts(
        calls,
        prior_counts=tool_attempt_counts(events, turn=turn),
    )

    async def _one(call: ToolCallDraft) -> dict[str, Any]:
        await store.append(
            session_id,
            EventDraft(
                event_type="tool/call",
                turn=turn,
                step=step,
                run_id=run_id,
                data={"call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            ),
        )
        definition = runtime.resolve(call.name)
        if abort.is_set():
            result = error_result("cancelled")
        elif definition is None:
            result = unknown_tool_result(call.name)
        elif call.call_id in blocked_ids:
            result = retry_exhausted_result(call.name)
        else:
            result = dict(await runtime.pipeline.run(definition, call.arguments))
        result = enrich_tool_result(result)
        content = result.get("content")
        if not isinstance(content, str):
            content = json.dumps(result, ensure_ascii=False)
        await store.append(
            session_id,
            EventDraft(
                event_type="tool/result",
                turn=turn,
                step=step,
                run_id=run_id,
                surface_op="append",
                data={
                    "call_id": call.call_id,
                    "name": call.name,
                    "ok": bool(result.get("ok", True)),
                    "content": content if isinstance(content, str) else str(content),
                    "evidence_id": result.get("evidence_id"),
                    "error": result.get("error"),
                    "error_class": result.get("error_class"),
                },
            ),
        )
        return dict(result)

    safe = all(
        (runtime.resolve(call.name) is not None and runtime.resolve(call.name).is_concurrency_safe)
        for call in calls
    )
    if safe and len(calls) > 1:
        gathered = await asyncio.gather(*[_one(call) for call in calls])
        return SchedulerOutcome(results=list(gathered))
    results = []
    for call in calls:
        results.append(await _one(call))
        if abort.is_set():
            break
    return SchedulerOutcome(results=results)
