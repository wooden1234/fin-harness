"""事实依据评测占位。"""

from __future__ import annotations


def judge_grounding(answer: str) -> dict:
    return {"answer_len": len(answer), "status": "not_configured"}
