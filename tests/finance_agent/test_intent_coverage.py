"""意图驱动 FinAgent：证据链解析、coverage gate、汇总降级。"""

from __future__ import annotations

import pytest
from langgraph.types import Send

from app.agents.finance_agent.join.fan_in import fan_in_ready, sub_task_satisfied
from app.agents.finance_agent.planner.dispatch_workers import (
    route_after_dispatch_workers,
    route_after_retrieval_worker,
)
from app.agents.finance_agent.planner.eval_cases import load_eval_cases, score_case
from app.agents.finance_agent.planner.resolve_evidence import (
    INTENT_TO_EVIDENCE_CHAIN,
    resolve_task_evidence,
)
from app.agents.finance_agent.planner.prompts import (
    PLANNER_REPAIR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from pathlib import Path
from app.agents.finance_agent.planner.validate import validate_and_normalize_tasks
from app.agents.states import SubTask


def test_intent_to_evidence_chain_mapping():
    assert INTENT_TO_EVIDENCE_CHAIN["product_policy"] == ["faq", "web_search"]
    assert INTENT_TO_EVIDENCE_CHAIN["structured_metric"] == ["financial_query", "web_search"]
    assert INTENT_TO_EVIDENCE_CHAIN["market_event"] == ["web_search"]


def test_resolve_task_evidence_fills_type_and_chain():
    task = SubTask(id="t1", question="信用卡年费怎么收", intent="product_policy")
    resolved = resolve_task_evidence(task)
    assert resolved.type == "faq"
    assert resolved.evidence_chain == ["faq", "web_search"]


def test_validate_accepts_intent_not_type():
    result = validate_and_normalize_tasks(
        [SubTask(id="t1", question="什么是 T+1", intent="concept_explain")]
    )
    assert len(result.tasks) == 1
    assert result.tasks[0].intent == "concept_explain"
    assert result.needs_repair is False


def test_validate_legacy_type_maps_to_intent():
    result = validate_and_normalize_tasks(
        [SubTask(id="t1", question="宁德时代 2024 年营业收入", type="financial_query")]
    )
    assert result.tasks[0].intent == "structured_metric"


def test_dispatch_sends_evidence_chain():
    tasks = [
        SubTask(
            id="t1",
            question="信用卡年费怎么收",
            intent="product_policy",
            type="faq",
            evidence_chain=["faq", "web_search"],
        )
    ]
    sends = route_after_dispatch_workers({"sub_tasks": tasks})
    assert len(sends) == 1
    assert sends[0].node == "faq_agent"
    assert sends[0].arg["evidence_chain"] == ["faq", "web_search"]


def test_coverage_gate_faq_uncovered_routes_to_web():
    route = route_after_retrieval_worker(
        {
            "sub_task_id": "t1",
            "sub_question": "信用卡年费怎么收",
            "evidence_chain": ["faq", "web_search"],
            "task_results": [
                {
                    "sub_task_id": "t1",
                    "type": "faq",
                    "coverage": "uncovered",
                }
            ],
        }
    )
    assert isinstance(route, Send)
    assert route.node == "web_search_agent"


def test_coverage_gate_sql_uncovered_routes_to_web_not_faq():
    route = route_after_retrieval_worker(
        {
            "sub_task_id": "t1",
            "sub_question": "宁德时代 2024 年营业收入",
            "evidence_chain": ["financial_query", "web_search"],
            "task_results": [
                {
                    "sub_task_id": "t1",
                    "type": "financial_query",
                    "coverage": "uncovered",
                }
            ],
        }
    )
    assert isinstance(route, Send)
    assert route.node == "web_search_agent"


def test_fan_in_waits_for_sql_then_web_chain():
    tasks = [
        SubTask(
            id="t1",
            question="宁德时代 2024 年营业收入",
            intent="structured_metric",
            type="financial_query",
            evidence_chain=["financial_query", "web_search"],
        )
    ]
    assert not fan_in_ready(
        sub_tasks=tasks,
        task_results=[
            {
                "sub_task_id": "t1",
                "type": "financial_query",
                "coverage": "uncovered",
            }
        ],
    )
    assert fan_in_ready(
        sub_tasks=tasks,
        task_results=[
            {
                "sub_task_id": "t1",
                "type": "financial_query",
                "coverage": "uncovered",
            },
            {
                "sub_task_id": "t1",
                "type": "web_search",
                "coverage": "covered",
            },
        ],
    )


def test_fan_in_sql_uncovered_chain_exhausted_is_ready():
    tasks = [
        SubTask(
            id="t1",
            question="宁德时代 2024 年毛利率",
            intent="structured_metric",
            type="financial_query",
            evidence_chain=["financial_query", "web_search"],
        )
    ]
    assert sub_task_satisfied(
        "t1",
        [
            {"sub_task_id": "t1", "type": "financial_query", "coverage": "uncovered"},
            {"sub_task_id": "t1", "type": "web_search", "coverage": "uncovered"},
        ],
        chain=["financial_query", "web_search"],
    )


def test_market_event_dispatches_web_only():
    task = resolve_task_evidence(
        SubTask(id="w1", question="最近证监会程序化交易新规", intent="market_event")
    )
    sends = route_after_dispatch_workers({"sub_tasks": [task]})
    assert sends[0].node == "web_search_agent"


def test_phase1_intent_eval_cases_load():
    path = Path(__file__).resolve().parents[2] / "knowledge" / "eval" / "intent_coverage_eval.jsonl"
    cases = load_eval_cases(path)
    assert len(cases) == 4
    product = next(c for c in cases if c["id"] == "phase1_product_policy")
    assert product["expected_intents"] == ["product_policy"]
    row = score_case(product, ["faq"], actual_intents=["product_policy"])
    assert row["intent_ok"] and row["type_ok"]


def test_planner_prompt_keeps_boundary_rules_compact():
    assert len(PLANNER_SYSTEM_PROMPT) <= 5000
    assert len(PLANNER_REPAIR_SYSTEM_PROMPT) <= 1000
    assert "不要按“答案是数字”判断" in PLANNER_SYSTEM_PROMPT
    assert "按单项计提坏账准备" in PLANNER_SYSTEM_PROMPT


def test_planner_eval_covers_pdf_sql_and_faq_boundaries():
    path = Path(__file__).resolve().parents[2] / "knowledge" / "eval" / "planner_eval.jsonl"
    cases = {case["id"]: case for case in load_eval_cases(path)}

    assert cases["boundary_pdf_bad_debt_table"]["expected_intents"] == ["document_qa"]
    assert cases["boundary_sql_canonical_metric"]["expected_intents"] == [
        "structured_metric"
    ]
    assert cases["boundary_faq_internal_policy"]["expected_intents"] == [
        "product_policy"
    ]
    assert cases["boundary_mixed_sql_pdf_detail"]["expected_task_count"] == 2
