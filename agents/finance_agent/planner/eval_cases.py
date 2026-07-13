"""Planner 评测用例加载与打分（供脚本与单测共用）。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_EVAL_TYPES = frozenset({"faq", "pdf", "financial_query", "web_search"})
# eval_cases.py → planner → finance_agent → agents → app → repo
DEFAULT_EVAL_PATH = (
    Path(__file__).resolve().parents[4] / "knowledge" / "eval" / "planner_eval.jsonl"
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
    required = ("id", "query", "expect_empty", "expected_types", "expected_task_count")
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(f"line {line_no}: missing fields {missing}")
    if not isinstance(case["query"], str) or not case["query"].strip():
        raise ValueError(f"line {line_no}: query must be non-empty string")
    if not isinstance(case["expected_types"], list):
        raise ValueError(f"line {line_no}: expected_types must be list")
    for t in case["expected_types"]:
        if t not in ALLOWED_EVAL_TYPES:
            raise ValueError(f"line {line_no}: invalid expected type {t!r}")
    if bool(case["expect_empty"]) != (int(case["expected_task_count"]) == 0):
        raise ValueError(f"line {line_no}: expect_empty inconsistent with expected_task_count")
    if bool(case["expect_empty"]) and case["expected_types"]:
        raise ValueError(f"line {line_no}: empty case must have expected_types=[]")


def score_case(case: dict[str, Any], actual_types: list[str]) -> dict[str, Any]:
    expect_empty = bool(case["expect_empty"])
    actual_empty = len(actual_types) == 0
    return {
        "id": case["id"],
        "query": case["query"],
        "expect_empty": expect_empty,
        "expected_types": case["expected_types"],
        "actual_types": actual_types,
        "empty_ok": expect_empty == actual_empty,
        "count_ok": len(actual_types) == int(case["expected_task_count"]),
        "type_ok": Counter(actual_types) == Counter(case["expected_types"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    type_ok = sum(1 for r in rows if r["type_ok"])
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
        "empty_accuracy": empty_ok / total if total else 0.0,
        "count_accuracy": count_ok / total if total else 0.0,
        "type_ok": type_ok,
        "empty_ok": empty_ok,
        "count_ok": count_ok,
        "triple_total": triple_total,
        "triple_type_accuracy": triple_type_ok / triple_total if triple_total else 0.0,
        "triple_count_accuracy": triple_count_ok / triple_total if triple_total else 0.0,
        "triple_type_ok": triple_type_ok,
        "triple_count_ok": triple_count_ok,
    }
