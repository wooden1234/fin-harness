"""输出前合规审查。"""

from __future__ import annotations

from compliance.policies import ComplianceDecision
from compliance.rules import find_rule_violations


def review_answer(answer: str) -> ComplianceDecision:
    violations = find_rule_violations(answer)
    if violations:
        return ComplianceDecision(
            action="block",
            reason_code="forbidden_investment_expression",
            reason=f"命中合规禁用表达：{', '.join(violations)}",
        )
    return ComplianceDecision(action="pass")
