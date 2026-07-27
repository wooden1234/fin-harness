"""CandidateSet 的确定性市场数据计算。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from agents.orchestrator.contracts import CandidateSet, MarketFilter, MarketQueryPlan


class MarketComputationError(ValueError):
    """市场查询计划无法安全执行。"""


def _field_value(row: Mapping[str, Any], field: str) -> Any:
    """支持使用点号读取嵌套字段。"""
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _has_field(row: Mapping[str, Any], field: str) -> bool:
    """判断记录是否声明了字段；字段值为 None 仍视为字段存在。"""
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


def _as_sequence(value: Any, *, operator: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise MarketComputationError(f"{operator}_requires_sequence")


def _matches(row: Mapping[str, Any], condition: MarketFilter) -> bool:
    actual = _field_value(row, condition.field)
    expected = condition.value
    operator = condition.operator

    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if actual is None:
        return False
    if operator == "in":
        return actual in _as_sequence(expected, operator=operator)
    if operator == "not_in":
        return actual not in _as_sequence(expected, operator=operator)
    if operator == "contains":
        try:
            return expected in actual
        except TypeError:
            return False
    if operator == "between":
        bounds = _as_sequence(expected, operator=operator)
        if len(bounds) != 2:
            raise MarketComputationError("between_requires_two_bounds")
        try:
            return bounds[0] <= actual <= bounds[1]
        except TypeError:
            return False

    comparisons = {
        "gt": lambda: actual > expected,
        "gte": lambda: actual >= expected,
        "lt": lambda: actual < expected,
        "lte": lambda: actual <= expected,
    }
    try:
        return comparisons[operator]()
    except (KeyError, TypeError):
        return False


def _sort_key(value: Any) -> tuple[int, int, Any]:
    """把空值和不同基础类型收敛为可比较的排序键。"""
    if value is None:
        return (1, 0, "")
    if isinstance(value, bool):
        return (0, 0, int(value))
    if isinstance(value, (int, float)):
        return (0, 1, value)
    return (0, 2, str(value))


def compute_candidate_set(
    source: CandidateSet,
    plan: MarketQueryPlan,
) -> CandidateSet:
    """按已校验计划过滤、排序并截取候选数据。"""
    if plan.universe != source.universe:
        raise MarketComputationError(
            f"universe_mismatch:{source.universe}:{plan.universe}"
        )
    if plan.enrichments:
        raise MarketComputationError("enrichments_require_provider_tools")
    if plan.group_by or plan.metrics:
        raise MarketComputationError("aggregation_not_supported")

    if source.rows:
        referenced_fields = [
            *[condition.field for condition in plan.filters],
            *[sort.field for sort in plan.sort],
        ]
        known_fields = {
            field
            for row in source.rows
            for field in referenced_fields
            if _has_field(row, field)
        }
        unknown_fields = sorted(set(referenced_fields) - known_fields)
        if unknown_fields:
            raise MarketComputationError(
                f"unknown_market_field:{','.join(unknown_fields)}"
            )

    rows = [
        dict(row)
        for row in source.rows
        if all(_matches(row, condition) for condition in plan.filters)
    ]
    for sort in reversed(plan.sort):
        rows.sort(
            key=lambda row, field=sort.field: _sort_key(_field_value(row, field)),
            reverse=sort.direction == "desc",
        )
        # 无论升降序都把缺失值放到末尾，避免空值占据 Top N。
        rows.sort(key=lambda row, field=sort.field: _field_value(row, field) is None)
    rows = rows[: plan.limit]

    plan_json = plan.model_dump_json()
    digest = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()[:12]
    metadata = dict(source.metadata)
    metadata.update(
        {
            "operation": "market_compute",
            "source_dataset_id": source.dataset_id,
            "source_row_count": len(source.rows),
            "result_row_count": len(rows),
        }
    )
    return CandidateSet(
        dataset_id=f"{source.dataset_id}:computed:{digest}",
        universe=source.universe,
        provider=source.provider,
        as_of=plan.as_of or source.as_of,
        rows=rows,
        query_plan=plan,
        evidence_ids=list(source.evidence_ids),
        metadata=metadata,
    )


__all__ = ["MarketComputationError", "compute_candidate_set"]
