"""引用评测占位。"""

from __future__ import annotations


def judge_citations(citations: list[dict]) -> dict:
    return {"citation_count": len(citations), "passed": bool(citations)}
