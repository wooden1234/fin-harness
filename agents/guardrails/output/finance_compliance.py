"""金融回答输出合规适配层。"""

from __future__ import annotations

from compliance.policies import ComplianceDecision
from compliance.review import review_answer


def review_finance_answer(answer: str) -> ComplianceDecision:
    """复用现有合规规则审查金融回答，避免维护重复策略。"""
    return review_answer(answer)


__all__ = ["review_finance_answer"]
