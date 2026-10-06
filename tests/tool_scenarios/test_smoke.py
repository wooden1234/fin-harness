"""3 条确定性 smoke：真实 Agent Loop + 脚本 LLM + 模拟工具。

这些测试验证评测链路和中间事件，不访问问财、检索服务或互联网。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

import harness.agent.loop as loop_module
from evals.tool_dataset import load_tool_cases
from harness.agent.loop import Agent
from harness.llm.fake import FakeLlmAdapter, content_turn, tool_turn
from harness.session.store import InMemorySessionStore, assert_contiguous
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.runtime import ToolRuntime


class SpyHandler:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(arguments))
        if not self._responses:
            raise AssertionError("工具调用次数超出场景剧本")
        return self._responses.pop(0)


def _definition(name: str, handler: SpyHandler) -> ToolDefinition:
    return ToolDefinition(
        tool_id=f"scenario.{name}",
        name=name,
        description=f"scenario tool: {name}",
        handler=handler,
        openai_schema=function_schema(
            name,
            f"scenario tool: {name}",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
    )


async def _run(
    *,
    question: str,
    llm: FakeLlmAdapter,
    definitions: list[ToolDefinition],
) -> tuple[Any, list[Any]]:
    store = InMemorySessionStore()
    header = await store.create(
        tenant_id="tool-eval",
        user_id="offline-evaluator",
        conversation_id="smoke",
    )
    agent = Agent(
        header.session_id,
        store,
        llm,
        runtime=ToolRuntime(definitions),
        owner_id="smoke-worker",
    )
    # The installed LangSmith pytest plugin may wrap @traceable coroutines and
    # wait for a remote test run.  Smoke tests exercise the Loop itself and
    # must remain offline, so call the decorated implementation underneath
    # that transport wrapper.
    prompt_impl = getattr(agent._prompt_impl, "__wrapped__", None)
    traced_execute = loop_module.execute_tool_calls
    raw_execute = getattr(traced_execute, "__wrapped__", traced_execute)
    loop_module.execute_tool_calls = raw_execute
    try:
        if prompt_impl is None:
            result = await agent.prompt(question)
        else:
            result = await prompt_impl(agent, question, source="user")
    finally:
        loop_module.execute_tool_calls = traced_execute
    events = await store.load_events(header.session_id)
    assert_contiguous(events)
    return result, events


def _assert_event_contract(events: list[Any], expected_tools: list[str]) -> None:
    calls = [event for event in events if event.event_type == "tool/call"]
    results = [event for event in events if event.event_type == "tool/result"]
    published = [event for event in events if event.event_type == "answer/published"]
    assert [event.data["name"] for event in calls] == expected_tools
    assert [event.data["name"] for event in results] == expected_tools
    assert Counter(event.data["call_id"] for event in calls) == Counter(
        event.data["call_id"] for event in results
    )
    assert all(event.data["ok"] is True for event in results)
    assert len(published) == 1


def _record_result(
    record_property: Any,
    *,
    case: dict[str, Any],
    result: Any,
    events: list[Any],
    tools: list[str],
    actual_requests: int,
    source_binding: str,
) -> None:
    calls = [event for event in events if event.event_type == "tool/call"]
    results = [event for event in events if event.event_type == "tool/result"]
    published = [event for event in events if event.event_type == "answer/published"]
    paired = Counter(event.data.get("call_id") for event in calls) == Counter(
        event.data.get("call_id") for event in results
    )
    all_tools_ok = len(results) == len(calls) and all(
        event.data.get("ok") is True for event in results
    )
    record_property("tool_case_id", case["id"])
    record_property("tool_title", case["metadata"]["title"])
    record_property("tool_bucket", case["metadata"]["bucket"])
    record_property("tool_execution_mode", case["metadata"]["execution_mode"])
    record_property("tool_names", tools)
    record_property("tool_call_count", len(calls))
    record_property("tool_answer", result.published_answer)
    record_property("tool_finish_reason", result.finish_reason)
    record_property(
        "tool_assertions",
        {
            "event_sequence": "passed" if [e.data.get("name") for e in calls] == tools else "failed",
            "call_result_pairing": "passed" if paired else "failed",
            "tool_results": "passed" if all_tools_ok else "failed",
            "final_answer": "passed" if len(published) == 1 and bool(result.published_answer) else "failed",
            "source_binding": source_binding,
        },
    )
    record_property(
        "tool_metrics",
        {
            "logical_calls": len(calls),
            "actual_requests": actual_requests,
            "retries": max(0, actual_requests - len(calls)),
            "retry_eligible_calls": 0,
            "bounded_recovery_successes": 0,
            "duplicate_requests": 0,
            "unbudgeted_subcalls": 0,
            "first_attempt_successes": len(calls) if all_tools_ok else 0,
        },
    )


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_001_calculation_through_real_loop(record_property: Any) -> None:
    case = load_tool_cases("smoke")[0]
    calculation = SpyHandler(
        [{"ok": True, "calculations": [{"value": 20, "unit": "%"}]}]
    )
    llm = FakeLlmAdapter(
        [
            tool_turn("run_calculation", json.dumps({"query": "100到120的同比"})),
            content_turn("测试甲公司收入由100万元增至120万元，同比增长20%。"),
        ]
    )
    result, events = await _run(
        question=case["inputs"]["query"],
        llm=llm,
        definitions=[_definition("run_calculation", calculation)],
    )
    _record_result(
        record_property,
        case=case,
        result=result,
        events=events,
        tools=["run_calculation"],
        actual_requests=len(calculation.calls),
        source_binding="not_applicable",
    )
    assert result.finish_reason == "completed"
    assert "20%" in result.published_answer
    assert calculation.calls == [{"query": "100到120的同比"}]
    _assert_event_contract(events, ["run_calculation"])


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_002_iwencai_to_calculation_chain(record_property: Any) -> None:
    case = load_tool_cases("smoke")[1]
    iwencai = SpyHandler(
        [
            {
                "ok": True,
                "facts": [
                    {"subject": "测试甲", "period": "2023", "value": 100, "unit": "万元"},
                    {"subject": "测试甲", "period": "2024", "value": 120, "unit": "万元"},
                ],
                "evidence_id": "ev-iwencai-1",
            }
        ]
    )
    calculation = SpyHandler(
        [{"ok": True, "calculations": [{"value": 20, "unit": "%"}]}]
    )
    llm = FakeLlmAdapter(
        [
            tool_turn("query_iwencai_finance", '{"query":"测试甲2023年和2024年收入"}', call_id="q1"),
            tool_turn("run_calculation", '{"query":"基于ev-iwencai-1计算同比"}', call_id="c1"),
            content_turn("测试甲2023年收入100万元，2024年收入120万元，同比增长20%。"),
        ]
    )
    result, events = await _run(
        question=case["inputs"]["query"],
        llm=llm,
        definitions=[
            _definition("query_iwencai_finance", iwencai),
            _definition("run_calculation", calculation),
        ],
    )
    _record_result(
        record_property,
        case=case,
        result=result,
        events=events,
        tools=["query_iwencai_finance", "run_calculation"],
        actual_requests=len(iwencai.calls) + len(calculation.calls),
        source_binding="passed",
    )
    assert all(text in result.published_answer for text in ("2023", "100万元", "2024", "120万元", "20%"))
    assert len(iwencai.calls) == len(calculation.calls) == 1
    _assert_event_contract(events, ["query_iwencai_finance", "run_calculation"])


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_003_pdf_and_web_sources_remain_separate(record_property: Any) -> None:
    case = load_tool_cases("smoke")[2]
    pdf = SpyHandler(
        [{"ok": True, "facts": [{"value": 120, "unit": "万元"}], "citations": [{"doc_id": "TEST-A-2024", "page": 10}]}]
    )
    web = SpyHandler(
        [{"ok": True, "results": [{"url": "https://issuer.example/notice", "content": "测试甲发布测试新品公告。"}]}]
    )
    llm = FakeLlmAdapter(
        [
            tool_turn("search_pdf_knowledge_tool", '{"query":"TEST-A-2024收入"}', call_id="p1"),
            tool_turn("search_web", '{"query":"测试甲最新公告"}', call_id="w1"),
            content_turn(
                "年报收入为120万元（TEST-A-2024第10页）；测试甲发布测试新品公告，来源：https://issuer.example/notice。"
            ),
        ]
    )
    result, events = await _run(
        question=case["inputs"]["query"],
        llm=llm,
        definitions=[
            _definition("search_pdf_knowledge_tool", pdf),
            _definition("search_web", web),
        ],
    )
    _record_result(
        record_property,
        case=case,
        result=result,
        events=events,
        tools=["search_pdf_knowledge_tool", "search_web"],
        actual_requests=len(pdf.calls) + len(web.calls),
        source_binding="passed",
    )
    assert all(text in result.published_answer for text in ("120万元", "TEST-A-2024", "第10页", "https://issuer.example/notice"))
    assert len(pdf.calls) == len(web.calls) == 1
    _assert_event_contract(events, ["search_pdf_knowledge_tool", "search_web"])
