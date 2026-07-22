"""长期记忆召回质量与成本指标（进程内聚合，后续可接 Prometheus）。"""

from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any

_lock = Lock()
_counters: Counter[str] = Counter()
_values: dict[str, float] = {}


def increment(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += value


def observe(name: str, value: float) -> None:
    with _lock:
        _values[name] = _values.get(name, 0.0) + value


def snapshot() -> dict[str, Any]:
    with _lock:
        counters = dict(_counters)
        totals = dict(_values)
        requests = counters.get("memory_recall_requests_total", 0)
        candidates = counters.get("memory_recall_candidates_total", 0)
        hits = counters.get("memory_recall_hits_total", 0)
        return {
            "counters": counters,
            "totals": totals,
            "rates": {
                "hit_rate": hits / requests if requests else 0.0,
                "store_fallback_rate": (
                    counters.get("memory_recall_sql_fallback_total", 0) / requests
                    if requests
                    else 0.0
                ),
                "avg_tokens_per_request": (
                    totals.get("memory_recall_tokens_total", 0.0) / requests
                    if requests
                    else 0.0
                ),
                "avg_recall_latency_ms": (
                    totals.get("memory_recall_latency_ms_total", 0.0) / requests
                    if requests
                    else 0.0
                ),
                "avg_embedding_latency_ms": (
                    totals.get("memory_embedding_latency_ms_total", 0.0)
                    / counters.get("memory_embedding_requests_total", 1)
                ),
            },
        }
