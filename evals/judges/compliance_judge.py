"""合规评测占位。"""

from __future__ import annotations

from compliance.review import review_answer


def judge_compliance(answer: str) -> dict:
    decision = review_answer(answer)
    return {"passed": decision.passed, "reason": decision.reason}
