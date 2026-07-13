"""回放评测入口。"""

from __future__ import annotations


def evaluate_replay(trace_id: str) -> dict:
    return {"trace_id": trace_id, "status": "not_configured"}
