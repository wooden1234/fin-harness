"""Agent 单 session 句柄。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from harness.agent.leases import InMemoryLeaseStore
from harness.agent.result import RunResult
from harness.approval.service import validate_decision
from harness.compaction.compact import maybe_compact
from harness.compaction.policy import compact_limit, policy_from_settings
from harness.contracts.errors import InvariantError, LlmError
from harness.finalization.submit import execute_submit_answer
from harness.llm.types import StreamAssembler
from harness.prompt.assembler import assemble_system, header_snapshot
from harness.prompt.preferences import load_preference_context
from harness.prompt.sections import default_sections, preference_section
from harness.session.invariant import assert_model_request_logged
from harness.session.store import SessionStore
from harness.session.surface import messages_for_llm, project_inbox
from harness.session.types import EventDraft, SessionEvent, new_id
from harness.tools.runtime import ToolRuntime
from harness.tools.scheduler import execute_tool_calls
from harness.tools.skill import inject_skill_context
from harness.tools.memory import memory_tool_definitions
from harness.tools.todo import todo_write_definition
from harness.tools.retry_policy import (
    USER_UNAVAILABLE_HINT,
    failed_twice_without_success,
    should_publish_unavailable,
)

_MAX_STEPS = 30


class Agent:
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        llm: Any,
        *,
        runtime: ToolRuntime | None = None,
        leases: Any | None = None,
        owner_id: str = "local",
    ) -> None:
        self.session_id = session_id
        self._store = store
        self._llm = llm
        self._runtime = runtime or ToolRuntime.builtin()
        self._leases = leases or InMemoryLeaseStore()
        self._owner_id = owner_id
        self._abort = asyncio.Event()
        self._compact_policy = policy_from_settings()
        self._compact_token_limit = compact_limit(self._compact_policy)
        self._published: str | None = None
        self._follow_ups: list[str] = []
        self._waiting: dict[str, Any] | None = None

    async def prompt(self, text: str, *, source: str = "user") -> RunResult:
        self._abort = asyncio.Event()
        self._published = None
        self._follow_ups = []
        self._waiting = None
        lease = await self._leases.acquire(self.session_id, self._owner_id)
        try:
            events = await self._store.load_events(self.session_id)
            turn = _next_turn(events)
            run_id = new_id()
            await self._store.append(
                self.session_id,
                EventDraft(event_type="turn/start", turn=turn, run_id=run_id, data={"turn": turn}),
            )
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="user/message",
                    turn=turn,
                    run_id=run_id,
                    surface_op="append",
                    data={"content": text, "source": source},
                ),
            )
            await self._claim_inbox(turn=turn, run_id=run_id)
            reason = await self._steps(turn=turn, run_id=run_id, start_step=1)
            if reason != "waiting_approval":
                await self._close_turn(turn, run_id, reason)
            return await self._result(run_id, reason)
        finally:
            await self._leases.release(self.session_id, lease.token)

    async def inject(self, text: str, *, source: str = "plugin") -> None:
        events = await self._store.load_events(self.session_id)
        turn = _current_turn(events) or 1
        await self._store.append(
            self.session_id,
            EventDraft(
                event_type="user/message",
                turn=turn,
                surface_op="append",
                data={"content": text, "source": source},
            ),
        )

    async def resume_approval(self, approval_id: str, *, decision: str = "allow") -> RunResult:
        self._abort = asyncio.Event()
        self._published = None
        self._follow_ups = []
        self._waiting = None
        lease = await self._leases.acquire(self.session_id, self._owner_id)
        try:
            events = await self._store.load_events(self.session_id)
            asked = validate_decision(events, approval_id)
            turn = int(next(event.turn for event in reversed(events) if event.turn))
            step = int(next((event.step or 1) for event in reversed(events) if event.step))
            run_id = next((event.run_id for event in reversed(events) if event.run_id), new_id())
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="approval/decided",
                    turn=turn,
                    step=step,
                    run_id=run_id,
                    data={"approval_id": approval_id, "decision": decision},
                ),
            )
            if decision != "allow":
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="tool/result",
                        turn=turn,
                        step=step,
                        run_id=run_id,
                        surface_op="append",
                        data={
                            "call_id": asked.get("call_id"),
                            "name": asked.get("name"),
                            "ok": False,
                            "content": json.dumps({"ok": False, "error": "denied"}),
                        },
                    ),
                )
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="step/end",
                        turn=turn,
                        step=step,
                        run_id=run_id,
                        data={"turn": turn, "step": step},
                    ),
                )
                reason = await self._steps(turn=turn, run_id=run_id, start_step=step + 1)
                if reason != "waiting_approval":
                    await self._close_turn(turn, run_id, reason)
                return await self._result(run_id, reason)
            runtime = self._bound_runtime(turn, run_id)
            arguments = asked.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            except json.JSONDecodeError:
                parsed = {}
            definition = runtime.resolve(str(asked.get("name") or ""))
            if definition is None:
                result: dict[str, Any] = {"ok": False, "error": "unknown_tool"}
            else:
                result = dict(await runtime.pipeline.run(definition, parsed if isinstance(parsed, dict) else {}))
            content = result.get("content")
            if not isinstance(content, str):
                content = json.dumps(result, ensure_ascii=False)
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="tool/result",
                    turn=turn,
                    step=step,
                    run_id=run_id,
                    surface_op="append",
                    data={
                        "call_id": asked.get("call_id"),
                        "name": asked.get("name"),
                        "ok": bool(result.get("ok", True)),
                        "content": content,
                        "evidence_id": result.get("evidence_id"),
                    },
                ),
            )
            if str(asked.get("name") or "") == "skill":
                await inject_skill_context(self._store, self.session_id, result, turn=turn, run_id=run_id)
            await self._store.append(
                self.session_id,
                EventDraft(event_type="step/end", turn=turn, step=step, run_id=run_id, data={"turn": turn, "step": step}),
            )
            reason = await self._steps(turn=turn, run_id=run_id, start_step=step + 1)
            if reason != "waiting_approval":
                await self._close_turn(turn, run_id, reason)
            return await self._result(run_id, reason)
        finally:
            await self._leases.release(self.session_id, lease.token)

    def cancel(self) -> None:
        self._abort.set()

    async def _claim_inbox(self, *, turn: int, run_id: str) -> None:
        events = await self._store.load_events(self.session_id)
        for item in project_inbox(events):
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="inbox/claimed",
                    turn=turn,
                    run_id=run_id,
                    data={"inbox_seq": item["seq"]},
                ),
            )
            content = str((item.get("data") or {}).get("content") or "").strip()
            if not content:
                continue
            source = str((item.get("data") or {}).get("source") or "inject")
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="user/message",
                    turn=turn,
                    run_id=run_id,
                    surface_op="append",
                    data={"content": content, "source": source},
                ),
            )

    def _bound_runtime(self, turn: int, run_id: str) -> ToolRuntime:
        async def _submit(arguments: dict[str, Any]) -> dict[str, Any]:
            events = await self._store.load_events(self.session_id)
            result = execute_submit_answer(arguments, events=events, turn=turn)
            if result.get("published"):
                self._published = str(result.get("markdown") or "")
                self._follow_ups = list(result.get("follow_ups") or [])
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="answer/published",
                        turn=turn,
                        run_id=run_id,
                        data={
                            "markdown": self._published,
                            "follow_ups": self._follow_ups,
                            "mode": result.get("mode"),
                        },
                    ),
                )
            return result

        extra = [
            todo_write_definition(self._store, self.session_id, turn=turn, run_id=run_id),
            *memory_tool_definitions(self._store, self.session_id, run_id=run_id),
        ]
        return self._runtime.rebind_submit(_submit).with_extra(extra)

    async def _publish_unavailable(self, *, turn: int, run_id: str) -> None:
        if self._published:
            return
        events = await self._store.load_events(self.session_id)
        result = execute_submit_answer(
            {
                "mode": "direct",
                "direct_answer": USER_UNAVAILABLE_HINT,
                "follow_ups": [],
            },
            events=events,
            turn=turn,
        )
        if not result.get("published"):
            return
        self._published = str(result.get("markdown") or USER_UNAVAILABLE_HINT)
        self._follow_ups = list(result.get("follow_ups") or [])
        await self._store.append(
            self.session_id,
            EventDraft(
                event_type="answer/published",
                turn=turn,
                run_id=run_id,
                data={
                    "markdown": self._published,
                    "follow_ups": self._follow_ups,
                    "mode": "direct",
                },
            ),
        )

    async def _maybe_give_up_without_tools(
        self,
        *,
        results: Sequence[Any],
        turn: int,
        run_id: str,
    ) -> None:
        events = await self._store.load_events(self.session_id)
        if should_publish_unavailable(results, events=events, turn=turn):
            await self._publish_unavailable(turn=turn, run_id=run_id)
            return
        if failed_twice_without_success(events, turn=turn):
            await self.inject(
                "同一工具已重试一次仍失败。若没有更匹配的工具或技能，请立即 "
                f"submit_answer（mode=direct）回复用户：{USER_UNAVAILABLE_HINT}",
                source="plugin",
            )

    async def _steps(self, *, turn: int, run_id: str, start_step: int) -> str:
        reminded = False
        overflow_retries = 0
        for step in range(start_step, _MAX_STEPS + 1):
            if self._abort.is_set():
                return "cancelled"
            if self._published:
                return "completed"
            await maybe_compact(
                store=self._store,
                session_id=self.session_id,
                llm=self._llm,
                turn=turn,
                run_id=run_id,
                token_limit=self._compact_token_limit,
                trigger="pressure",
                policy=self._compact_policy,
            )
            await self._store.append(
                self.session_id,
                EventDraft(event_type="step/start", turn=turn, step=step, run_id=run_id, data={"turn": turn, "step": step}),
            )
            try:
                assembled = await self._model_step(turn=turn, step=step, run_id=run_id)
            except InvariantError as exc:
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="invariant/violation",
                        turn=turn,
                        step=step,
                        run_id=run_id,
                        data={"code": exc.code, "error": str(exc)},
                    ),
                )
                await self._store.append(
                    self.session_id,
                    EventDraft(event_type="step/end", turn=turn, step=step, run_id=run_id, data={"error": exc.code}),
                )
                return "error"
            except LlmError as exc:
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="step/end",
                        turn=turn,
                        step=step,
                        run_id=run_id,
                        data={"error": exc.code, "message": str(exc)[:800]},
                    ),
                )
                if exc.code == "cancelled":
                    return "cancelled"
                if exc.code == "context_overflow" and overflow_retries < self._compact_policy.max_overflow_retries:
                    overflow_retries += 1
                    await maybe_compact(
                        store=self._store,
                        session_id=self.session_id,
                        turn=turn,
                        run_id=run_id,
                        token_limit=self._compact_token_limit,
                        trigger="context-overflow",
                        policy=self._compact_policy,
                    )
                    continue
                return "error"
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="assistant/message",
                    turn=turn,
                    step=step,
                    run_id=run_id,
                    surface_op="append",
                    data={
                        "content": assembled.content,
                        "tool_calls": [
                            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                            for call in assembled.tool_calls
                        ],
                    },
                ),
            )
            if not assembled.tool_calls:
                await self._store.append(
                    self.session_id,
                    EventDraft(event_type="step/end", turn=turn, step=step, run_id=run_id, data={"turn": turn, "step": step}),
                )
                if not reminded:
                    reminded = True
                    await self.inject("必须调用 submit_answer 才能结束本轮。", source="plugin")
                    continue
                return "error"
            runtime = self._bound_runtime(turn, run_id)
            outcome = await execute_tool_calls(
                store=self._store,
                session_id=self.session_id,
                runtime=runtime,
                calls=assembled.tool_calls,
                turn=turn,
                step=step,
                run_id=run_id,
                abort=self._abort,
            )
            if outcome.waiting_approval:
                self._waiting = {
                    "approval_id": outcome.approval_id,
                    "call_id": outcome.call_id,
                    "name": outcome.name,
                }
                return "waiting_approval"
            for call, result in zip(assembled.tool_calls, outcome.results or []):
                if call.name == "skill" and result.get("ok"):
                    await inject_skill_context(self._store, self.session_id, result, turn=turn, run_id=run_id)
            await self._maybe_give_up_without_tools(
                results=outcome.results or [],
                turn=turn,
                run_id=run_id,
            )
            await self._store.append(
                self.session_id,
                EventDraft(event_type="step/end", turn=turn, step=step, run_id=run_id, data={"turn": turn, "step": step}),
            )
            if self._published:
                return "completed"
        if not self._published:
            await self.inject("必须调用 submit_answer 才能结束本轮。", source="plugin")
            extra = _MAX_STEPS + 1
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="step/start",
                    turn=turn,
                    step=extra,
                    run_id=run_id,
                    data={"step": extra, "turn": turn},
                ),
            )
            try:
                assembled = await self._model_step(turn=turn, step=extra, run_id=run_id)
            except (InvariantError, LlmError):
                return "error"
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="assistant/message",
                    turn=turn,
                    step=extra,
                    run_id=run_id,
                    surface_op="append",
                    data={
                        "content": assembled.content,
                        "tool_calls": [
                            {
                                "call_id": call.call_id,
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                            for call in assembled.tool_calls
                        ],
                    },
                ),
            )
            if assembled.tool_calls:
                runtime = self._bound_runtime(turn, run_id)
                extra_outcome = await execute_tool_calls(
                    store=self._store,
                    session_id=self.session_id,
                    runtime=runtime,
                    calls=assembled.tool_calls,
                    turn=turn,
                    step=extra,
                    run_id=run_id,
                    abort=self._abort,
                )
                await self._maybe_give_up_without_tools(
                    results=extra_outcome.results or [],
                    turn=turn,
                    run_id=run_id,
                )
            await self._store.append(
                self.session_id,
                EventDraft(
                    event_type="step/end",
                    turn=turn,
                    step=extra,
                    run_id=run_id,
                    data={"step": extra, "turn": turn},
                ),
            )
            if self._published:
                return "completed"
        return "error"

    async def _model_step(self, *, turn: int, step: int, run_id: str):
        events = await self._store.load_events(self.session_id)
        loaded = await load_preference_context(
            store=self._store,
            session_id=self.session_id,
            events=events,
            turn=turn,
        )
        sections = list(default_sections())
        pref = preference_section(loaded.preferences, loaded.turn_overrides)
        if pref is not None:
            sections.append(pref)
        system = assemble_system(sections)
        runtime = self._bound_runtime(turn, run_id)
        tools = runtime.openai_tools()
        header = header_snapshot(
            system=system,
            tools=tools,
            adapter_defaults=dict(getattr(self._llm, "adapter_defaults", {}) or {}),
        )
        await self._store.append(
            self.session_id,
            EventDraft(
                event_type="request/header",
                turn=turn,
                step=step,
                run_id=run_id,
                data=header,
            ),
        )
        events = await self._store.load_events(self.session_id)
        assert_model_request_logged(events, turn=turn, step=step)
        assembler = StreamAssembler()
        async for chunk in self._llm.stream(
            system=system,
            messages=messages_for_llm(events),
            tools=tools,
            abort=self._abort,
        ):
            assembler.push(chunk)
            if chunk.kind == "content" and chunk.text:
                await self._store.append(
                    self.session_id,
                    EventDraft(
                        event_type="assistant/chunk",
                        turn=turn,
                        step=step,
                        run_id=run_id,
                        data={"text": chunk.text},
                    ),
                )
        return assembler.finalize()

    async def _close_turn(self, turn: int, run_id: str, reason: str) -> None:
        events = await self._store.load_events(self.session_id)
        if any(event.event_type == "turn/end" and event.turn == turn for event in events):
            return
        await self._store.append(
            self.session_id,
            EventDraft(
                event_type="turn/end",
                turn=turn,
                run_id=run_id,
                data={"turn": turn, "reason": reason},
            ),
        )

    async def _result(self, run_id: str, reason: str) -> RunResult:
        events = await self._store.load_events(self.session_id)
        waiting = self._waiting or {}
        return RunResult(
            session_id=self.session_id,
            run_id=run_id,
            finish_reason=reason,
            published_answer=self._published,
            follow_ups=self._follow_ups,
            events=events,
            waiting_approval=reason == "waiting_approval",
            approval_id=waiting.get("approval_id"),
            call_id=waiting.get("call_id"),
            error=None if reason in {"completed", "waiting_approval", "cancelled"} else reason,
        )


def _next_turn(events: Sequence[SessionEvent]) -> int:
    turns = [event.turn for event in events if event.turn is not None]
    return (max(turns) if turns else 0) + 1


def _current_turn(events: Sequence[SessionEvent]) -> int | None:
    turns = [event.turn for event in events if event.turn is not None]
    return max(turns) if turns else None
