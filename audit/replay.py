"""审计回放占位。"""

from __future__ import annotations


def replay_run(trace_id: str) -> dict:
    return {"trace_id": trace_id, "status": "not_configured"}
