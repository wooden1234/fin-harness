"""工具调度：可并行的一起跑，结果按模型顺序提交。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from harness.llm.types import ToolCallDraft
from harness.session.types import EventDraft, SessionEvent, new_id
from harness.tools.arguments import coerce_tool_arguments
from harness.tools.definition import ToolDefinition
from harness.tools.errors import enrich_tool_result, error_result
from harness.tools.pipeline import ToolPipeline
from harness.tools.retry_policy import (
    allocate_tool_attempts,
    blocked_retries,
    retry_exhausted_result,
    tool_attempt_counts,
    unknown_tool_result,
)
from harness.tools.todo import (
    TODO_PLAN_APPROVAL_MESSAGE,
    TODO_PLAN_DECLINED_GUIDANCE,
    TODO_PLAN_PENDING_GUIDANCE,
    TODO_PLAN_REQUIRED_GUIDANCE,
    todo_plan_state,
    user_wants_plan_first,
)
from harness.tracing import traceable


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
    side_results: list[tuple[str, dict[str, Any]]] | None = None


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    payloads = coerce_tool_arguments(raw)
    if not payloads:
        text = str(raw or "").strip()
        return {} if not text else None
    return payloads[0]


def _tool_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    calls = inputs.get("calls") or []
    return {
        "session_id": inputs.get("session_id"),
        "turn": inputs.get("turn"),
        "step": inputs.get("step"),
        "tools": [getattr(call, "name", None) for call in calls],
    }


def _tool_result_content(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(result, ensure_ascii=False)


async def _append_tool_call(
    store: Any,
    *,
    session_id: str,
    call: ToolCallDraft,
    turn: int,
    step: int,
    run_id: str,
) -> None:
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


async def _append_tool_result(
    store: Any,
    *,
    session_id: str,
    call: ToolCallDraft,
    result: Mapping[str, Any],
    turn: int,
    step: int,
    run_id: str,
) -> None:
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
                "content": _tool_result_content(result),
                "evidence_id": result.get("evidence_id"),
                "error": result.get("error"),
                "error_class": result.get("error_class"),
            },
        ),
    )


async def _run_recorded_tool(
    runtime: ToolResolver,
    call: ToolCallDraft,
    *,
    abort: asyncio.Event,
    policy_blocks: Mapping[str, dict[str, Any]],
    blocked_ids: set[str],
) -> dict[str, Any]:
    definition = runtime.resolve(call.name)
    if abort.is_set():
        return error_result("cancelled")
    if definition is None:
        return unknown_tool_result(call.name)
    if call.call_id in policy_blocks:
        return dict(policy_blocks[call.call_id])
    if call.call_id in blocked_ids:
        return retry_exhausted_result(call.name)
    return dict(await runtime.pipeline.run(definition, call.arguments))


async def run_parked_calls(
    *,
    store: Any,
    session_id: str,
    runtime: ToolResolver,
    calls: Sequence[ToolCallDraft],
    turn: int,
    step: int,
    run_id: str,
    abort: asyncio.Event,
) -> list[tuple[str, dict[str, Any]]]:
    """用户确认之后再执行先前按下的查数调用。"""
    results: list[tuple[str, dict[str, Any]]] = []
    for call in calls:
        await _append_tool_call(
            store, session_id=session_id, call=call, turn=turn, step=step, run_id=run_id
        )
        result = enrich_tool_result(
            await _run_recorded_tool(
                runtime,
                call,
                abort=abort,
                policy_blocks={},
                blocked_ids=set(),
            )
        )
        await _append_tool_result(
            store,
            session_id=session_id,
            call=call,
            result=result,
            turn=turn,
            step=step,
            run_id=run_id,
        )
        results.append((call.name, result))
    return results


@traceable(name="agent.tools", run_type="tool", process_inputs=_tool_trace_inputs)
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
    events = await store.load_events(session_id)
    blocked_ids, _counts = allocate_tool_attempts(
        calls,
        prior_counts=tool_attempt_counts(events, turn=turn),
    )
    policy_blocks = blocked_retries(calls, events=events, turn=turn)
    plan_state = todo_plan_state(events, turn=turn)
    if plan_state == "declined":
        for call in calls:
            if call.name != "todo_write" or call.call_id in policy_blocks:
                continue
            policy_blocks[call.call_id] = error_result(
                "todo_plan_declined",
                name=call.name,
                model_guidance=TODO_PLAN_DECLINED_GUIDANCE,
            )
    approval_call = None
    for call in calls:
        definition = runtime.resolve(call.name)
        if definition is None or call.call_id in blocked_ids or call.call_id in policy_blocks:
            continue
        needs_approval = definition.requires_human_approval
        if call.name == "todo_write" and plan_state == "ask":
            needs_approval = True
        if needs_approval:
            approval_call = call
            break
    if approval_call is not None and approval_call.name != "todo_write":
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
            deferred = policy_blocks.get(call.call_id)
            if deferred is None and call.call_id in blocked_ids:
                deferred = retry_exhausted_result(call.name)
            if deferred is None:
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
                        "error": deferred.get("error"),
                        "error_class": deferred.get("error_class"),
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

    if approval_call is not None and approval_call.name == "todo_write":
        approval_id = new_id()
        pending_calls: list[dict[str, str]] = []
        for call in calls:
            if call.call_id == approval_call.call_id:
                await _append_tool_call(
                    store, session_id=session_id, call=call, turn=turn, step=step, run_id=run_id
                )
                continue
            sibling = runtime.resolve(call.name)
            if call.name == "todo_write":
                result = error_result(
                    "todo_plan_pending",
                    name=call.name,
                    model_guidance=TODO_PLAN_PENDING_GUIDANCE,
                )
            elif sibling is not None and sibling.requires_human_approval:
                result = error_result("deferred_for_approval", name=call.name)
            else:
                pending_calls.append(
                    {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                )
                continue
            await _append_tool_call(
                store, session_id=session_id, call=call, turn=turn, step=step, run_id=run_id
            )
            result = enrich_tool_result(result)
            await _append_tool_result(
                store,
                session_id=session_id,
                call=call,
                result=result,
                turn=turn,
                step=step,
                run_id=run_id,
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
                    "message": TODO_PLAN_APPROVAL_MESSAGE,
                    "pending_calls": pending_calls,
                },
            ),
        )
        return SchedulerOutcome(
            waiting_approval=True,
            approval_id=approval_id,
            call_id=approval_call.call_id,
            name=approval_call.name,
        )

    if (
        plan_state == "ask"
        and user_wants_plan_first(events, turn=turn)
        and approval_call is None
    ):
        results = []
        for call in calls:
            await _append_tool_call(
                store, session_id=session_id, call=call, turn=turn, step=step, run_id=run_id
            )
            result = enrich_tool_result(
                error_result(
                    "todo_plan_required",
                    name=call.name,
                    model_guidance=TODO_PLAN_REQUIRED_GUIDANCE,
                )
            )
            await _append_tool_result(
                store,
                session_id=session_id,
                call=call,
                result=result,
                turn=turn,
                step=step,
                run_id=run_id,
            )
            results.append(result)
        return SchedulerOutcome(results=results)

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
        result = enrich_tool_result(
            await _run_recorded_tool(
                runtime,
                call,
                abort=abort,
                policy_blocks=policy_blocks,
                blocked_ids=blocked_ids,
            )
        )
        await _append_tool_result(
            store,
            session_id=session_id,
            call=call,
            result=result,
            turn=turn,
            step=step,
            run_id=run_id,
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
