"""Redis 客户端进程内指标。"""

from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any

_lock = Lock()
_counters: Counter[str] = Counter()
_latency_ms_total = 0.0


def record_initialization(*, success: bool) -> None:
    with _lock:
        _counters[
            "redis_initialization_success_total"
            if success
            else "redis_initialization_error_total"
        ] += 1


def record_operation(
    operation: str,
    *,
    success: bool,
    latency_ms: float,
    retries: int,
) -> None:
    global _latency_ms_total
    safe_operation = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in operation.lower()
    )
    with _lock:
        _counters["redis_operations_total"] += 1
        _counters[f"redis_operation_{safe_operation}_total"] += 1
        _counters[
            "redis_operation_success_total"
            if success
            else "redis_operation_error_total"
        ] += 1
        _counters["redis_retries_total"] += max(0, retries)
        _latency_ms_total += max(0.0, latency_ms)


def snapshot() -> dict[str, Any]:
    with _lock:
        counters = dict(_counters)
        operations = counters.get("redis_operations_total", 0)
        latency_total = _latency_ms_total
    return {
        "counters": counters,
        "totals": {
            "redis_operation_latency_ms_total": latency_total,
        },
        "averages": {
            "redis_operation_latency_ms": (
                latency_total / operations if operations else 0.0
            ),
        },
    }


def reset_metrics() -> None:
    """仅供测试隔离进程内指标。"""
    global _latency_ms_total
    with _lock:
        _counters.clear()
        _latency_ms_total = 0.0


__all__ = [
    "record_initialization",
    "record_operation",
    "reset_metrics",
    "snapshot",
]
