"""Planner 评测用例加载与打分（供脚本与单测共用）。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_EVAL_TYPES = frozenset({"faq", "pdf", "financial_query", "web_search"})
ALLOWED_EVAL_INTENTS = frozenset(
    {
        "concept_explain",
        "product_policy",
        "document_qa",
        "structured_metric",
        "market_event",
    }
)

# 意图 → 首选证据工具（与 resolve_evidence 映射一致）
INTENT_TO_PRIMARY_TOOL = {
    "concept_explain": "faq",
    "product_policy": "faq",
    "document_qa": "pdf",
    "structured_metric": "financial_query",
    "market_event": "web_search",
}

# eval_cases.py → planner → finance_agent → agents → app → repo
DEFAULT_EVAL_PATH = (
    Path(__file__).resolve().parents[3] / "knowledge" / "eval" / "planner_eval.jsonl"
)


def load_eval_cases(path: Path | None = None) -> list[dict[str, Any]]:
    eval_path = path or DEFAULT_EVAL_PATH
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        validate_eval_case(case, line_no=line_no)
        cases.append(case)
    return cases


def validate_eval_case(case: dict[str, Any], *, line_no: int) -> None:
    required = ("id", "query", "expect_empty", "expected_task_count")
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(f"line {line_no}: missing fields {missing}")
    if not isinstance(case["query"], str) or not case["query"].strip():
        raise ValueError(f"line {line_no}: query must be non-empty string")

    has_intents = "expected_intents" in case
    has_types = "expected_types" in case
    if not has_intents and not has_types:
        raise ValueError(f"line {line_no}: need expected_intents or expected_types")

    if has_intents:
        if not isinstance(case["expected_intents"], list):
            raise ValueError(f"line {line_no}: expected_intents must be list")
        for intent in case["expected_intents"]:
            if intent not in ALLOWED_EVAL_INTENTS:
                raise ValueError(f"line {line_no}: invalid expected intent {intent!r}")

    if has_types:
        if not isinstance(case["expected_types"], list):
            raise ValueError(f"line {line_no}: expected_types must be list")
        for t in case["expected_types"]:
            if t not in ALLOWED_EVAL_TYPES:
                raise ValueError(f"line {line_no}: invalid expected type {t!r}")

    if bool(case["expect_empty"]) != (int(case["expected_task_count"]) == 0):
        raise ValueError(f"line {line_no}: expect_empty inconsistent with expected_task_count")
    if bool(case["expect_empty"]):
        if has_intents and case["expected_intents"]:
            raise ValueError(f"line {line_no}: empty case must have expected_intents=[]")
        if has_types and case["expected_types"]:
            raise ValueError(f"line {line_no}: empty case must have expected_types=[]")


def expected_intents_for_case(case: dict[str, Any]) -> list[str]:
    if case.get("expected_intents") is not None:
        return list(case["expected_intents"])
    return []


def expected_types_for_case(case: dict[str, Any]) -> list[str]:
    if case.get("expected_types") is not None:
        return list(case["expected_types"])
    return [INTENT_TO_PRIMARY_TOOL[i] for i in case.get("expected_intents") or []]


def score_case(
    case: dict[str, Any],
    actual_types: list[str],
    *,
    actual_intents: list[str] | None = None,
) -> dict[str, Any]:
    expect_empty = bool(case["expect_empty"])
    actual_empty = len(actual_types) == 0
    expected_types = expected_types_for_case(case)
    expected_intents = expected_intents_for_case(case)
    intents = actual_intents if actual_intents is not None else []

    return {
        "id": case["id"],
        "query": case["query"],
        "expect_empty": expect_empty,
        "expected_types": expected_types,
        "expected_intents": expected_intents,
        "actual_types": actual_types,
        "actual_intents": intents,
        "empty_ok": expect_empty == actual_empty,
        "count_ok": len(actual_types) == int(case["expected_task_count"]),
        "type_ok": Counter(actual_types) == Counter(expected_types),
        "intent_ok": (
            Counter(intents) == Counter(expected_intents)
            if expected_intents
            else True
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    type_ok = sum(1 for r in rows if r["type_ok"])
    intent_ok = sum(1 for r in rows if r.get("intent_ok", True))
    empty_ok = sum(1 for r in rows if r["empty_ok"])
    count_ok = sum(1 for r in rows if r["count_ok"])

    triple_rows = [
        r for r in rows if len(r.get("expected_types") or []) == 3
    ]
    triple_total = len(triple_rows)
    triple_type_ok = sum(1 for r in triple_rows if r["type_ok"])
    triple_count_ok = sum(1 for r in triple_rows if r["count_ok"])

    return {
        "total": total,
        "type_accuracy": type_ok / total if total else 0.0,
        "intent_accuracy": intent_ok / total if total else 0.0,
        "empty_accuracy": empty_ok / total if total else 0.0,
        "count_accuracy": count_ok / total if total else 0.0,
        "type_ok": type_ok,
        "intent_ok": intent_ok,
        "empty_ok": empty_ok,
        "count_ok": count_ok,
        "triple_total": triple_total,
        "triple_type_accuracy": triple_type_ok / triple_total if triple_total else 0.0,
        "triple_count_accuracy": triple_count_ok / triple_total if triple_total else 0.0,
        "triple_type_ok": triple_type_ok,
        "triple_count_ok": triple_count_ok,
    }
