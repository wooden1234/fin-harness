"""给工具结果盖 evidence_id。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4


def stamp_evidence(tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("ok", True)
    out.setdefault("evidence_id", f"{tool_id}:{uuid4().hex[:12]}")
    out.setdefault("source_tool", tool_id)
    return out
