"""Run evaluation cases through the real Harness Agent and project event traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from uuid import uuid4

# The production backend package lives below app/backend; make the standalone
# ``python -m evals...`` entrypoints behave like the documented uvicorn command.
_BACKEND_DIR = Path(__file__).resolve().parents[1] / "app" / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from harness.control.manager import AgentManager
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime


def _json_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _collect_named(value: Any, names: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                if isinstance(item, dict):
                    found.append(dict(item))
                elif isinstance(item, list):
                    found.extend(dict(entry) for entry in item if isinstance(entry, dict))
            found.extend(_collect_named(item, names))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_named(item, names))
    return found


def project_run_events(events: Iterable[Any], *, run_id: str) -> dict[str, Any]:
    selected = [event for event in events if str(getattr(event, "run_id", "") or "") == run_id]
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    claim_links: list[dict[str, Any]] = []
    for event in selected:
        data = dict(getattr(event, "data", None) or {})
        if event.event_type == "tool/call":
            calls.append(data)
        elif event.event_type == "tool/result":
            payload = _json_payload(data.get("content"))
            result = {**data, "payload": payload}
            results.append(result)
            facts.extend(_collect_named(payload, {"facts"}))
            citations.extend(_collect_named(payload, {"citations", "evidence"}))
            claims.extend(_collect_named(payload, {"claims"}))
            claim_links.extend(_collect_named(payload, {"claim_evidence_links"}))

    started = min((event.created_at for event in selected), default=None)
    ended = max((event.created_at for event in selected), default=None)
    event_latency = (
        max(0.0, (ended - started).total_seconds() * 1000)
        if started is not None and ended is not None
        else None
    )
    return {
        "tool_calls": calls,
        "tool_results": results,
        "facts": facts,
        "citations": citations,
        "claims": claims,
        "claim_evidence_links": claim_links,
        "event_latency_ms": event_latency,
    }


class HarnessEvalAdapter:
    """One adapter instance owns an isolated in-memory session store per experiment."""

    def __init__(
        self,
        *,
        manager: AgentManager | None = None,
        store: InMemorySessionStore | None = None,
        llm: Any | None = None,
        runtime: ToolRuntime | None = None,
    ) -> None:
        if manager is not None:
            self.manager = manager
            self.store = manager.store
        else:
            self.store = store or InMemorySessionStore()
            self.manager = AgentManager(
                store=self.store,
                llm=llm,
                runtime=runtime or ToolRuntime.product(),
            )

    async def run_case(self, inputs: dict[str, Any], *, case_id: str = "") -> dict[str, Any]:
        conversation_id = f"eval-{case_id or uuid4()}-{uuid4()}"
        agent = await self.manager.get(
            tenant_id="eval",
            user_id="offline-evaluator",
            conversation_id=conversation_id,
            owner_id="offline-evaluator",
        )
        turns = inputs.get("turns")
        if isinstance(turns, list):
            user_messages = [
                str(item.get("content") or "")
                for item in turns
                if isinstance(item, dict)
                and item.get("role") == "user"
                and str(item.get("content") or "").strip()
            ]
        else:
            query = str(inputs.get("query") or "").strip()
            user_messages = [query] if query else []
        if not user_messages:
            raise ValueError("evaluation input requires query or user turns")

        started = perf_counter()
        turn_outputs: list[dict[str, Any]] = []
        all_calls: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        all_facts: list[dict[str, Any]] = []
        all_citations: list[dict[str, Any]] = []
        all_claims: list[dict[str, Any]] = []
        all_links: list[dict[str, Any]] = []
        final_result = None
        for text in user_messages:
            result = await agent.prompt(text)
            final_result = result
            trace = project_run_events(result.events, run_id=result.run_id)
            turn_output = {
                "query": text,
                "answer": result.published_answer or "",
                "finish_reason": result.finish_reason,
                "error": result.error,
                "waiting_approval": result.waiting_approval,
                "run_id": result.run_id,
                **trace,
            }
            turn_outputs.append(turn_output)
            all_calls.extend(trace["tool_calls"])
            all_results.extend(trace["tool_results"])
            all_facts.extend(trace["facts"])
            all_citations.extend(trace["citations"])
            all_claims.extend(trace["claims"])
            all_links.extend(trace["claim_evidence_links"])
            if result.waiting_approval or result.error:
                break

        assert final_result is not None
        return {
            "answer": final_result.published_answer or "",
            "finish_reason": final_result.finish_reason,
            "error": final_result.error,
            "waiting_approval": final_result.waiting_approval,
            "session_id": final_result.session_id,
            "run_id": final_result.run_id,
            "conversation_id": conversation_id,
            "turns": turn_outputs,
            "tool_calls": all_calls,
            "tool_results": all_results,
            "facts": all_facts,
            "citations": all_citations,
            "claims": all_claims,
            "claim_evidence_links": all_links,
            "latency_ms": (perf_counter() - started) * 1000,
        }


__all__ = ["HarnessEvalAdapter", "project_run_events"]
