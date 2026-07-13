"""Planner 校验、失败分级与评测集门禁。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Overwrite

from app.agents.finance_agent.planner.eval_cases import (
    DEFAULT_EVAL_PATH,
    load_eval_cases,
    score_case,
    summarize,
)
from app.agents.finance_agent.planner.common import (
    is_schema_error,
    is_transient_api_error,
)
from app.agents.finance_agent.planner.node import (
    supervisor_node,
)
from app.agents.finance_agent.planner.plan_tasks import (
    plan_tasks_node,
)
from app.agents.finance_agent.planner.repair_plan import (
    repair_plan_node,
)
from app.agents.finance_agent.planner.validate_plan import (
    route_after_validate_plan,
    validate_plan_node,
)
from app.agents.finance_agent.planner.validate import (
    MAX_SUBTASKS,
    validate_and_normalize_tasks,
)
from app.agents.states import PlannerOutput, SubTask

EVAL_PATH = DEFAULT_EVAL_PATH


def test_validate_drops_empty_and_forbidden_types():
    unknown = SubTask.model_construct(id="3", question="未知类型问题", type="other")
    result = validate_and_normalize_tasks(
        [
            SubTask(id="1", question="  ", intent="concept_explain"),
            SubTask(id="2", question="随便问问", type="general"),
            unknown,
            SubTask(id="4", question="什么是 T+1", intent="concept_explain"),
        ]
    )
    assert len(result.tasks) == 1
    assert result.tasks[0].question == "什么是 T+1"
    assert result.tasks[0].intent == "concept_explain"
    assert result.needs_repair is True
    assert "empty_question" in result.issues
    assert "forbidden_type:general" in result.issues
    assert any(i.startswith("unknown_type:") for i in result.issues)


def test_validate_merges_near_duplicates_and_caps():
    tasks = [
        SubTask(id="1", question="宁德时代 2024 年营业收入", intent="structured_metric"),
        SubTask(id="2", question="宁德时代2024年营业收入", intent="structured_metric"),
        SubTask(id="3", question="问题A", intent="concept_explain"),
        SubTask(id="4", question="问题B", intent="document_qa"),
        SubTask(id="5", question="问题C", intent="market_event"),
        SubTask(id="6", question="问题D", intent="concept_explain"),
    ]
    result = validate_and_normalize_tasks(tasks)
    assert len(result.tasks) <= MAX_SUBTASKS
    assert any(i.startswith("merged_duplicate:") for i in result.issues)
    assert any(i.startswith("exceeds_max:") for i in result.issues)
    assert result.needs_repair is False


def test_transient_and_schema_error_detection():
    assert is_transient_api_error(TimeoutError("x"))
    assert is_transient_api_error(ConnectionError("x"))

    class RateLimitError(Exception):
        pass

    assert is_transient_api_error(RateLimitError("429"))

    class OutputParserException(Exception):
        pass

    assert is_schema_error(OutputParserException("bad json"))
    assert not is_schema_error(RuntimeError("boom"))


@pytest.mark.asyncio
async def test_supervisor_skip_empty_query_has_reason():
    out = await supervisor_node({"sub_tasks": []}, {})
    assert out["sub_tasks"] == []
    assert out["steps"] == ["validate_plan:skip"]
    assert out["planner_error_reason"] == "empty_query"
    assert isinstance(out["task_results"], Overwrite)


@pytest.mark.asyncio
async def test_supervisor_api_error_retries_then_fallback():
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))
    mock_llm.with_structured_output.return_value = mock_structured

    with patch(
        "agents.finance_agent.planner.common.get_router_llm",
        return_value=mock_llm,
    ):
        out = await supervisor_node(
            {"messages": [HumanMessage(content="宁德时代营收")]},
            {},
    )

    assert out["sub_tasks"] == []
    assert out["steps"] == ["validate_plan:skip"]
    assert out["planner_error_reason"] == "api_error"
    assert mock_structured.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_supervisor_schema_error_attempts_repair():
    class OutputParserException(Exception):
        pass

    good = PlannerOutput(tasks=[SubTask(id="t1", question="什么是T+1", type="faq")])
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        side_effect=[OutputParserException("bad json"), good]
    )
    mock_llm.with_structured_output.return_value = mock_structured

    with patch(
        "agents.finance_agent.planner.common.get_router_llm",
        return_value=mock_llm,
    ):
        out = await supervisor_node(
            {"messages": [HumanMessage(content="什么是 T+1？")]},
            {},
    )

    assert len(out["sub_tasks"]) == 1
    assert out["sub_tasks"][0].type == "faq"
    assert out["steps"] == ["repair_plan"]
    assert out["planner_repair_attempted"] is True
    assert mock_structured.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_supervisor_validation_repair_then_success():
    dirty = PlannerOutput(
        tasks=[
            SubTask(id="t1", question="", type="faq"),
            SubTask(id="t2", question="随便", type="general"),
        ]
    )
    fixed = PlannerOutput(
        tasks=[SubTask(id="t3", question="T+1 交易制度是什么意思", type="faq")]
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=[dirty, fixed])
    mock_llm.with_structured_output.return_value = mock_structured

    with patch(
        "agents.finance_agent.planner.common.get_router_llm",
        return_value=mock_llm,
    ):
        out = await supervisor_node(
            {"messages": [HumanMessage(content="什么是 T+1？")]},
            {},
        )

    assert len(out["sub_tasks"]) == 1
    assert out["sub_tasks"][0].type == "faq"
    assert mock_structured.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_supervisor_unclassifiable_empty_plan():
    mock_output = PlannerOutput(tasks=[])
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=mock_output)
    mock_llm.with_structured_output.return_value = mock_structured

    with patch(
        "agents.finance_agent.planner.common.get_router_llm",
        return_value=mock_llm,
    ):
        out = await supervisor_node(
            {"messages": [HumanMessage(content="帮我看看")]},
            {},
        )

    assert out["sub_tasks"] == []
    assert out["steps"] == ["validate_plan:unclassifiable"]
    assert out["planner_error_reason"] == "unclassifiable"


@pytest.mark.asyncio
async def test_explicit_planner_nodes_route_to_repair_then_dispatch():
    dirty = PlannerOutput(
        tasks=[
            SubTask(id="t1", question="", type="faq"),
            SubTask(id="t2", question="随便", type="general"),
        ]
    )
    fixed = PlannerOutput(
        tasks=[SubTask(id="t3", question="T+1 交易制度是什么意思", type="faq")]
    )
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=[dirty, fixed])
    mock_llm.with_structured_output.return_value = mock_structured
    state = {"messages": [HumanMessage(content="什么是 T+1？")]}

    with patch(
        "agents.finance_agent.planner.common.get_router_llm",
        return_value=mock_llm,
    ):
        planned = await plan_tasks_node(state, {})
        validated = await validate_plan_node({**state, **planned}, {})
        assert route_after_validate_plan(validated) == "repair_plan"
        repaired = await repair_plan_node({**state, **planned, **validated}, {})

    assert len(repaired["sub_tasks"]) == 1
    assert repaired["planner_repair_attempted"] is True
    assert route_after_validate_plan(repaired) == "resolve_evidence"


def test_planner_eval_dataset_loads_and_scores():
    cases = load_eval_cases(EVAL_PATH)
    assert len(cases) >= 10

    rows = [
        score_case(cases[0], ["faq"]),
        score_case(
            next(c for c in cases if c["expect_empty"]),
            [],
        ),
    ]
    summary = summarize(rows)
    assert summary["type_accuracy"] == 1.0
    assert summary["empty_accuracy"] == 1.0


def test_planner_eval_jsonl_is_well_formed():
    raw = EVAL_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert raw
    for line in raw:
        case = json.loads(line)
        assert case["id"]
        assert case["query"]
