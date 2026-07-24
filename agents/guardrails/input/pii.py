"""输入敏感身份信息规则。"""

from __future__ import annotations

import re

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailStage,
    allow_input,
)

PII_PATTERNS = (
    (r"\d{15}(\d{2}[0-9Xx])?", "身份证号", "pii.identity_number"),
    (r"\d{16}(\d{3})?", "银行卡号", "pii.bank_card"),
    (r"1[3-9]\d{9}", "手机号", "pii.phone_number"),
)


def check_pii(query: str) -> GuardrailDecision:
    """检查当前策略禁止进入模型的敏感身份信息。"""
    for pattern, pii_type, rule_id in PII_PATTERNS:
        if re.search(pattern, query):
            return GuardrailDecision(
                action=GuardrailAction.BLOCK,
                stage=GuardrailStage.INPUT,
                reason_code="pii_detected",
                reason=f"检测到{pii_type}",
                matched_rules=[rule_id],
                metadata={"pii_type": pii_type},
            )
    return allow_input()


__all__ = ["PII_PATTERNS", "check_pii"]
