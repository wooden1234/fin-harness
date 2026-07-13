"""来源评级。"""

from __future__ import annotations

from evidence.models import Evidence


def source_score(evidence: Evidence) -> float:
    if evidence.source_type in {"pdf", "financial_fact"}:
        return 0.9
    if evidence.url:
        return 0.7
    return 0.5
