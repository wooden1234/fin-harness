"""Agent 单 session 句柄。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from harness.agent.leases import InMemoryLeaseStore
from harness.agent.result import RunResult
from harness.control.services import AgentControl
from harness.control.policy import TurnPolicy
from harness.compaction.compact import maybe_compact
from harness.compaction.policy import compact_limit, policy_from_settings
from harness.contracts.errors import ContextBudgetExhaustedError, InvariantError, LlmError
from harness.llm.types import StreamAssembler
from harness.session.invariant import assert_model_request_logged
from harness.session.store import SessionStore
from harness.session.surface import messages_for_llm, project_inbox
from harness.session.types import EventDraft, SessionEvent, new_id
from harness.tools.runtime import ToolRuntime
from harness.tools.scheduler import execute_tool_calls
from harness.tools.retry_policy import USER_UNAVAILABLE_HINT
from harness.runtime.context import request_header

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
        self._control = AgentControl(store, session_id)
        self._turn_policy = TurnPolicy()
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
            asked = self._control.validate_approval(events, approval_id)
            turn = int(next(event.turn for event in reversed(events) if event.turn))
            step = int(next((event.step or 1) for event in reversed(events) if event.step))
            run_id = next((event.run_id for event in reversed(events) if event.run_id), new_id())
            runtime = self._bound_runtime(turn, run_id)
            result = await self._control.approvals.apply(
                store=self._store,
                session_id=self.session_id,
                asked=asked,
                approval_id=approval_id,
                decision=decision,
                turn=turn,
                step=step,
                run_id=run_id,
                runtime=runtime,
            )
            if not self._control.approvals.is_allowed(decision):
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
            if str(asked.get("name") or "") == "skill" and result.get("ok"):
                await self._control.inject_skill(result, turn=turn, run_id=run_id)
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
        return self._control.bind_runtime(
            turn=turn, run_id=run_id, base_runtime=self._runtime
        )

    async def _publish(self, markdown: str, *, turn: int, run_id: str) -> None:
        if self._published:
            return
        text = self._turn_policy.finalize(markdown)
        if not text:
            return
        self._published = text
        await self._store.append(
            self.session_id,
            EventDraft(
                event_type="answer/published",
                turn=turn,
                run_id=run_id,
                data={"markdown": self._published, "follow_ups": self._follow_ups},
            ),
        )

    async def _apply_tool_error_policy(
        self,
        *,
        results: Sequence[Any],
        turn: int,
        run_id: str,
    ) -> None:
        events = await self._store.load_events(self.session_id)
        decision = self._turn_policy.decide_after_tools(results, events=events, turn=turn)
        if decision.action == "publish":
            await self._publish(decision.text or USER_UNAVAILABLE_HINT, turn=turn, run_id=run_id)
            return
        if decision.action == "inject" and decision.text:
            await self.inject(decision.text, source="plugin")

    async def _steps(self, *, turn: int, run_id: str, start_step: int) -> str:
        overflow_retries = 0
        for step in range(start_step, _MAX_STEPS + 1):
            if self._abort.is_set():
                return "cancelled"
            if self._published:
                return "completed"
            await self._store.append(
                self.session_id,
                EventDraft(event_type="step/start", turn=turn, step=step, run_id=run_id, data={"turn": turn, "step": step}),
            )
            try:
                system, tools = await self._request_context(turn=turn, run_id=run_id)
                await maybe_compact(
                    store=self._store,
                    session_id=self.session_id,
                    llm=self._llm,
                    turn=turn,
                    run_id=run_id,
                    token_limit=self._compact_token_limit,
                    allow_llm=True,
                    trigger="pressure",
                    policy=self._compact_policy,
                    system=system,
                    tools=tools,
                    enforce_budget=True,
                )
                assembled = await self._model_step(
                    turn=turn, step=step, run_id=run_id, system=system, tools=tools
                )
            except ContextBudgetExhaustedError as exc:
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
                return "error"
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
                    try:
                        await maybe_compact(
                            store=self._store,
                            session_id=self.session_id,
                            llm=self._llm,
                            turn=turn,
                            run_id=run_id,
                            token_limit=self._compact_token_limit,
                            allow_llm=True,
                            trigger="context-overflow",
                            policy=self._compact_policy,
                            system=system,
                            tools=tools,
                            enforce_budget=True,
                        )
                    except ContextBudgetExhaustedError:
                        return "error"
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
                await self._publish(assembled.content, turn=turn, run_id=run_id)
                return "completed" if self._published else "error"
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
                    await self._control.inject_skill(result, turn=turn, run_id=run_id)
            await self._apply_tool_error_policy(
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
        return "error"

    async def _request_context(self, *, turn: int, run_id: str) -> tuple[str, list[dict[str, Any]]]:
        return await self._control.request_context(
            turn=turn, run_id=run_id, base_runtime=self._runtime
        )

    async def _model_step(
        self,
        *,
        turn: int,
        step: int,
        run_id: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        if system is None or tools is None:
            system, tools = await self._request_context(turn=turn, run_id=run_id)
        header = request_header(system=system, tools=tools, llm=self._llm)
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
