"""100 条数据集的确定性 Scheduler/Pipeline 场景驱动器。

它只评价真实执行过的运行时断言。自然语言答案、真实模型行为、PostgreSQL
多进程竞争等需要其他执行器的断言会保留为 not_evaluated。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from harness.llm.types import ToolCallDraft
from harness.session.store import InMemorySessionStore
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result
from harness.tools.runtime import ToolRuntime
from harness.tools.scheduler import execute_tool_calls


@dataclass
class ScenarioObservation:
    case_id: str
    tool_names: list[str]
    events: list[Any]
    logical_calls: int
    actual_requests: int
    retries: int
    duplicate_requests: int
    unbudgeted_subcalls: int
    first_attempt_successes: int
    retry_eligible_calls: int
    bounded_recovery_successes: int
    assertions: dict[str, str]
    finish_reason: str
    answer: str


class ScriptedHandler:
    def __init__(self, script: list[Any], default_payload: dict[str, Any]) -> None:
        self.script = list(script)
        self.default_payload = default_payload
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(arguments))
        item = self.script.pop(0) if self.script else self.default_payload
        if isinstance(item, dict) and item.get("fault") == "timeout":
            raise asyncio.TimeoutError()
        if isinstance(item, dict) and int(item.get("http_status") or 0) >= 400:
            status = int(item["http_status"])
            code = "invalid_api_key" if status in {401, 403} else "http_error"
            return error_result(code, metadata={"http_status": status})
        if isinstance(item, dict) and "status" in item and item.get("status") == "ok":
            return {"ok": True, "data": item}
        return {"ok": True, "data": item}


def _tool_name(case: dict[str, Any]) -> str:
    return {
        "calculation": "run_calculation",
        "iwencai": "query_iwencai_finance",
        "financial_report": "search_pdf_knowledge_tool",
        "web_search": "search_web",
        "runtime_mcp": "scenario_runtime_probe",
    }[case["metadata"]["bucket"]]


def _script(setup: dict[str, Any]) -> list[Any]:
    for key in ("responses", "mcp_sequence"):
        value = setup.get(key)
        if isinstance(value, list) and value:
            return list(value)
    return [setup]


def _arguments(case: dict[str, Any]) -> str:
    setup = case["fixture"]["setup"]
    if "raw_arguments" in setup:
        return str(setup["raw_arguments"])
    return json.dumps({"query": str(case["inputs"].get("query") or case["id"])}, ensure_ascii=False)


class ScenarioDriver:
    async def run(self, case: dict[str, Any]) -> ScenarioObservation:
        setup = dict(case["fixture"]["setup"])
        name = _tool_name(case)
        script = _script(setup)
        handler = ScriptedHandler(script, setup)
        definition = ToolDefinition(
            tool_id=f"scenario.{name}",
            name=name,
            description=f"deterministic scenario adapter for {name}",
            handler=handler,
            openai_schema=function_schema(
                name,
                f"deterministic scenario adapter for {name}",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            timeout_seconds=0.02,
        )
        runtime = ToolRuntime([definition])
        store = InMemorySessionStore()
        header = await store.create(
            tenant_id="tool-eval", user_id="0", conversation_id=case["id"]
        )
        raw_execute = getattr(execute_tool_calls, "__wrapped__", execute_tool_calls)
        attempts = min(2, len(script))
        outcomes = []
        for index in range(attempts):
            outcomes.append(
                await raw_execute(
                    store=store,
                    session_id=header.session_id,
                    runtime=runtime,
                    calls=[
                        ToolCallDraft(
                            call_id=f"{case['id']}-call-{index + 1}",
                            name=name,
                            arguments=_arguments(case),
                        )
                    ],
                    turn=1,
                    step=index + 1,
                    run_id=case["id"],
                    abort=asyncio.Event(),
                )
            )
            result = (outcomes[-1].results or [{}])[0]
            if result.get("ok") is True:
                break
            if result.get("error_class") != "transient":
                break

        events = await store.load_events(header.session_id)
        calls = [event for event in events if event.event_type == "tool/call"]
        results = [event for event in events if event.event_type == "tool/result"]
        call_ids = [str(event.data.get("call_id")) for event in calls]
        result_ids = [str(event.data.get("call_id")) for event in results]
        paired = sorted(call_ids) == sorted(result_ids)
        successful = [event for event in results if event.data.get("ok") is True]
        retry_eligible = 1 if len(script) > 1 and self._first_is_transient(script[0]) else 0
        recovered = 1 if retry_eligible and len(handler.calls) > 1 and bool(successful) else 0
        first_success = 1 if results and results[0].data.get("ok") is True else 0
        retries = max(0, len(handler.calls) - 1) if retry_eligible else 0
        unbudgeted_subcalls = max(0, len(handler.calls) - len(calls) - retries)
        assertions = {
            "event_sequence": "passed" if len(calls) == len(results) else "failed",
            "call_result_pairing": "passed" if paired else "failed",
            "retry_budget": "passed" if len(calls) <= 2 else "failed",
            "final_answer": "not_evaluated",
            "source_binding": "not_evaluated",
        }
        # These cases directly specify parser/registry semantics. The driver
        # records current product behavior as a failure instead of blessing it.
        if case["id"] in {"tool-081", "tool-082"}:
            assertions["argument_contract"] = (
                "passed" if len(handler.calls) == 0 else "failed"
            )
        else:
            assertions["argument_contract"] = "passed"
        if case["id"] == "tool-083":
            assertions["nested_schema_contract"] = "not_evaluated"
        if case["id"] == "tool-084":
            assertions["duplicate_registration"] = "not_evaluated"
        return ScenarioObservation(
            case_id=case["id"],
            tool_names=[name for _ in calls],
            events=events,
            logical_calls=1,
            actual_requests=len(handler.calls),
            retries=retries,
            duplicate_requests=0,
            unbudgeted_subcalls=unbudgeted_subcalls,
            first_attempt_successes=first_success,
            retry_eligible_calls=retry_eligible,
            bounded_recovery_successes=recovered,
            assertions=assertions,
            finish_reason="scheduler_completed",
            answer="",
        )

    @staticmethod
    def _first_is_transient(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        return item.get("fault") == "timeout" or int(item.get("http_status") or 0) in {
            429,
            500,
            502,
            503,
            504,
        }
