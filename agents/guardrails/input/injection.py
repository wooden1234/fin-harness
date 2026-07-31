"""输入提示词注入规则。"""

from __future__ import annotations

from dataclasses import dataclass
import re

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
    allow_input,
)
from agents.guardrails.input.normalization import NormalizedInput, ensure_normalized


@dataclass(frozen=True, slots=True)
class InjectionRule:
    """带稳定标识和风险分值的注入检测规则。"""

    rule_id: str
    pattern: str
    score: int


INJECTION_RULES = (
    InjectionRule(
        "injection.override_instructions.zh",
        r"忽略.*指令",
        70,
    ),
    InjectionRule(
        "injection.override_instructions.en",
        r"ignore.*instruction",
        70,
    ),
    InjectionRule("injection.dan_role", r"你.*现在.*是.*DAN", 75),
    InjectionRule(
        "injection.system_prompt_reference",
        r"system\s*prompt",
        45,
    ),
    InjectionRule("injection.im_start_token", r"<\|im_start\|>", 80),
    InjectionRule("injection.im_end_token", r"<\|im_end\|>", 80),
    InjectionRule("injection.system_marker", r"\[SYSTEM\]", 60),
    InjectionRule("injection.instruction_marker", r"\[INST\]", 60),
    InjectionRule(
        "injection.system_prompt_extraction",
        r"你的.*系统.*提示词",
        75,
    ),
    InjectionRule(
        "injection.forget_previous",
        r"忘记.*之前",
        65,
    ),
    InjectionRule("injection.role_switch.zh", r"扮演.*角色", 25),
    InjectionRule(
        "injection.role_switch.en",
        r"pretend.*you.*are",
        25,
    ),
)

INJECTION_BLOCK_THRESHOLD = 70
INJECTION_CLARIFY_THRESHOLD = 25

BLOCK_INJECTION_PATTERNS = tuple(
    rule.pattern
    for rule in INJECTION_RULES
    if rule.score >= INJECTION_BLOCK_THRESHOLD
)
CLARIFY_INJECTION_PATTERNS = tuple(
    rule.pattern
    for rule in INJECTION_RULES
    if rule.score < INJECTION_BLOCK_THRESHOLD
)
INJECTION_PATTERNS = tuple(rule.pattern for rule in INJECTION_RULES)


def check_injection(query: str | NormalizedInput) -> GuardrailDecision:
    """按累计风险分值判断提示词注入。"""
    normalized = ensure_normalized(query)
    findings: list[GuardrailFinding] = []
    matched_rules: list[str] = []
    total_score = 0

    for rule in INJECTION_RULES:
        if re.search(rule.pattern, normalized.compact, re.IGNORECASE):
            total_score += rule.score
            matched_rules.append(rule.rule_id)
            findings.append(
                GuardrailFinding(
                    rule_id=rule.rule_id,
                    category="prompt_injection",
                    severity=(
                        GuardrailSeverity.HIGH
                        if rule.score >= INJECTION_BLOCK_THRESHOLD
                        else GuardrailSeverity.MEDIUM
                    ),
                    message="检测到提示词注入风险特征",
                    metadata={"score": rule.score},
                )
            )

    if total_score < INJECTION_CLARIFY_THRESHOLD:
        return allow_input()

    if total_score >= INJECTION_BLOCK_THRESHOLD:
        return GuardrailDecision(
            action=GuardrailAction.BLOCK,
            stage=GuardrailStage.INPUT,
            severity=GuardrailSeverity.HIGH,
            reason_code="prompt_injection_detected",
            reason="检测到明确的提示词注入特征",
            findings=findings,
            matched_rules=matched_rules,
            metadata={
                "score": total_score,
                "block_threshold": INJECTION_BLOCK_THRESHOLD,
                "clarify_threshold": INJECTION_CLARIFY_THRESHOLD,
            },
        )

    return GuardrailDecision(
        action=GuardrailAction.CLARIFY,
        stage=GuardrailStage.INPUT,
        severity=GuardrailSeverity.MEDIUM,
        reason_code="prompt_injection_suspected",
        reason="请求包含需要澄清的角色切换内容",
        findings=findings,
        matched_rules=matched_rules,
        metadata={
            "score": total_score,
            "block_threshold": INJECTION_BLOCK_THRESHOLD,
            "clarify_threshold": INJECTION_CLARIFY_THRESHOLD,
        },
    )


__all__ = [
    "BLOCK_INJECTION_PATTERNS",
    "CLARIFY_INJECTION_PATTERNS",
    "INJECTION_BLOCK_THRESHOLD",
    "INJECTION_CLARIFY_THRESHOLD",
    "INJECTION_PATTERNS",
    "INJECTION_RULES",
    "InjectionRule",
    "check_injection",
]
