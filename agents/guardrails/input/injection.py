"""输入提示词注入规则。"""

from __future__ import annotations

import re

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailStage,
    allow_input,
)

INJECTION_PATTERNS = (
    r"忽略.*指令",
    r"ignore.*instruction",
    r"你.*现在.*是.*DAN",
    r"system\s*prompt",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"你的.*系统.*提示词",
    r"忘记.*之前",
    r"扮演.*角色",
    r"pretend.*you.*are",
)


def check_injection(query: str) -> GuardrailDecision:
    """检查直接提示词注入特征。"""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return GuardrailDecision(
                action=GuardrailAction.BLOCK,
                stage=GuardrailStage.INPUT,
                reason_code="prompt_injection_detected",
                reason="检测到提示词注入特征",
                matched_rules=[pattern],
            )
    return allow_input()


__all__ = ["INJECTION_PATTERNS", "check_injection"]
