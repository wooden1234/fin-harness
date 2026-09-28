"""Deterministic structured financial fact evaluator."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


_MONEY_FACTORS = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "亿元": Decimal("100000000"),
    "千元": Decimal("1000"),
    "百万元": Decimal("1000000"),
    "百万元人民币": Decimal("1000000"),
}
_PERCENT_UNITS = {"%", "百分比"}
_POINT_UNITS = {"百分点", "percentage_point", "percentage points"}


def _text(value: Any) -> str:
    return re.sub(r"[\s,，:：()（）\[\]]+", "", str(value or "")).casefold()


def _subject(fact: dict[str, Any]) -> str:
    return _text(
        fact.get("subject")
        or fact.get("company")
        or fact.get("entity")
        or fact.get("issuer")
    )


def _period(fact: dict[str, Any]) -> str:
    return _text(
        fact.get("period")
        or fact.get("period_year")
        or fact.get("year")
        or fact.get("report_period")
    ).replace("年度", "").replace("财年", "")


def _metric(fact: dict[str, Any]) -> str:
    return _text(fact.get("metric") or fact.get("metric_name") or fact.get("name"))


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return Decimal(str(value))
    text = str(value).strip().replace(",", "").replace("，", "")
    text = text.removesuffix("%").strip()
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def normalized_numeric(value: Any, unit: Any) -> tuple[Decimal | None, str]:
    number = _decimal(value)
    normalized_unit = str(unit or "").strip()
    if number is None:
        return None, normalized_unit
    if normalized_unit in _MONEY_FACTORS:
        return number * _MONEY_FACTORS[normalized_unit], "money_base"
    if normalized_unit in _PERCENT_UNITS:
        return number, "percent"
    if normalized_unit in _POINT_UNITS:
        return number, "percentage_point"
    return number, normalized_unit


def _same_identity(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_subject = _subject(expected)
    actual_subject = _subject(actual)
    if expected_subject and actual_subject and expected_subject not in actual_subject and actual_subject not in expected_subject:
        return False
    expected_period = _period(expected)
    actual_period = _period(actual)
    if expected_period and actual_period and expected_period != actual_period:
        return False
    expected_metric = _metric(expected)
    actual_metric = _metric(actual)
    return bool(expected_metric and actual_metric) and (
        expected_metric == actual_metric
        or expected_metric in actual_metric
        or actual_metric in expected_metric
    )


def _same_value(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_value = expected.get("value")
    actual_value = actual.get("value")
    if isinstance(expected_value, list):
        actual_items = actual_value if isinstance(actual_value, list) else [actual_value]
        actual_text = {_text(item) for item in actual_items}
        return all(any(_text(wanted) in item or item in _text(wanted) for item in actual_text) for wanted in expected_value)
    expected_number, expected_kind = normalized_numeric(expected_value, expected.get("unit"))
    actual_number, actual_kind = normalized_numeric(actual_value, actual.get("unit"))
    if expected_number is not None and actual_number is not None:
        if expected_kind != actual_kind:
            return False
        tolerance = Decimal(str(expected.get("tolerance", "0")))
        if tolerance == 0:
            tolerance = max(abs(expected_number) * Decimal("0.0001"), Decimal("0.000001"))
        comparison = str(expected.get("comparison") or "equal")
        if comparison == "greater_than":
            return actual_number > expected_number or abs(actual_number - expected_number) <= tolerance
        return abs(actual_number - expected_number) <= tolerance
    return _text(expected_value) == _text(actual_value)


def evaluate_facts(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = [item for item in reference_outputs.get("required_facts") or [] if isinstance(item, dict)]
    actual = [item for item in outputs.get("facts") or [] if isinstance(item, dict)]
    if not expected:
        return [
            {"key": "fact_coverage", "score": None, "comment": "No structured facts required."},
            {"key": "fact_accuracy", "score": None, "comment": "No structured facts required."},
        ]

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    used: set[int] = set()
    for wanted in expected:
        candidate_index = next(
            (
                index
                for index, item in enumerate(actual)
                if index not in used and _same_identity(wanted, item) and _same_value(wanted, item)
            ),
            None,
        )
        if candidate_index is None:
            missing.append(wanted)
        else:
            used.add(candidate_index)
            matched.append((wanted, actual[candidate_index]))

    coverage = len(matched) / len(expected)
    conflicts: list[str] = []
    for wanted in missing:
        wanted_metric = _metric(wanted)
        for item in actual:
            if wanted_metric and wanted_metric == _metric(item):
                if _subject(wanted) and _subject(item) and _subject(wanted) != _subject(item):
                    conflicts.append(f"wrong_subject:{item.get('subject') or item.get('company') or item.get('entity')}")
                if _period(wanted) and _period(item) and _period(wanted) != _period(item):
                    conflicts.append(f"wrong_period:{item.get('period') or item.get('period_year') or item.get('year')}")

    results = [
        {
            "key": "fact_coverage",
            "score": coverage,
            "comment": f"matched={len(matched)}/{len(expected)} missing={missing}",
        },
        {
            "key": "fact_accuracy",
            "score": 1.0 if coverage == 1.0 else 0.0,
            "comment": "Structured facts only; final answer text is not parsed into facts.",
        },
    ]
    if conflicts:
        results.append(
            {
                "key": "fact_hard_failure",
                "score": 0.0,
                "comment": ";".join(sorted(set(conflicts))),
            }
        )
    return results


__all__ = ["evaluate_facts", "normalized_numeric"]
