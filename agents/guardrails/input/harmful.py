"""输入敏感内容的确定性意图区分。"""

from __future__ import annotations

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

_HARMFUL_TERMS = {
    "自杀": "self_harm",
    "自残": "self_harm",
    "杀人": "violence",
    "爆炸": "explosives",
    "炸药": "explosives",
    "枪支": "weapons",
    "毒品": "drugs",
    "色情": "sexual_content",
}

_CRISIS_PATTERNS = (
    r"我(?:现在)?(?:真的)?(?:想|要|准备|打算)(?:去)?(?:自杀|自残|结束生命)",
    r"我(?:不想活|活不下去)了?",
    r"结束我自己的生命",
)

_INSTRUCTION_PATTERNS = (
    r"(?:如何|怎么|怎样)(?![^，。！？]{0,8}(?:预防|避免|阻止|识别|帮助|救助))"
    r"[^，。！？]{0,16}(?:自杀|自残|杀人)",
    r"(?:制作|制造|购买|获取|使用|引爆|贩卖)[^，。！？]{0,16}"
    r"(?:炸药|爆炸|枪支|毒品)",
    r"(?:炸药|爆炸|枪支|毒品)[^，。！？]{0,16}"
    r"(?:制作|制造|购买|获取|使用|引爆|贩卖)",
    r"(?:生成|提供|观看|下载)[^，。！？]{0,16}色情",
)

_BENIGN_CONTEXT_PATTERN = re.compile(
    r"预防|避免|阻止|识别|帮助|救助|风险|法规|监管|新闻|分析|影响|治理|"
    r"科普|保险|行业|公司|犯罪|研究|历史|统计|心理|治疗"
)


def _find_terms(text: str) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for term, category in _HARMFUL_TERMS.items():
        if term in text:
            categories.setdefault(category, []).append(term)
    return categories


def _build_findings(
    categories: dict[str, list[str]],
    severity: GuardrailSeverity,
    intent: str,
) -> list[GuardrailFinding]:
    return [
        GuardrailFinding(
            rule_id=f"harmful.{category}",
            category="harmful_content",
            severity=severity,
            message="检测到需要进行意图区分的敏感内容",
            metadata={
                "content_category": category,
                "intent": intent,
                "term_count": len(terms),
            },
        )
        for category, terms in categories.items()
    ]


def check_harmful(query: str | NormalizedInput) -> GuardrailDecision:
    """区分危机、实施请求、公开讨论和上下文不足的敏感内容。"""
    normalized = ensure_normalized(query)
    text = normalized.canonical
    categories = _find_terms(text)
    crisis_detected = any(
        re.search(pattern, text) for pattern in _CRISIS_PATTERNS
    )
    if crisis_detected and "self_harm" not in categories:
        categories["self_harm"] = ["crisis_expression"]
    if not categories:
        return allow_input()

    if crisis_detected:
        severity = GuardrailSeverity.CRITICAL
        intent = "crisis"
        action = GuardrailAction.ESCALATE
        reason_code = "self_harm_crisis_detected"
        reason = "检测到可能需要立即支持的自伤风险"
    elif any(re.search(pattern, text) for pattern in _INSTRUCTION_PATTERNS):
        severity = GuardrailSeverity.HIGH
        intent = "instructions"
        action = GuardrailAction.BLOCK
        reason_code = "harmful_instructions_detected"
        reason = "检测到高风险行为实施请求"
    elif _BENIGN_CONTEXT_PATTERN.search(text):
        severity = GuardrailSeverity.LOW
        intent = "informational"
        action = GuardrailAction.ALLOW
        reason_code = "harmful_content_contextual"
        reason = "敏感词用于公开信息、风险或研究讨论"
    else:
        severity = GuardrailSeverity.MEDIUM
        intent = "ambiguous"
        action = GuardrailAction.CLARIFY
        reason_code = "harmful_content_needs_context"
        reason = "敏感内容缺少足够上下文"

    findings = _build_findings(categories, severity, intent)
    return GuardrailDecision(
        action=action,
        stage=GuardrailStage.INPUT,
        severity=severity,
        reason_code=reason_code,
        reason=reason,
        findings=findings,
        matched_rules=[finding.rule_id for finding in findings],
        metadata={
            "intent": intent,
            "content_categories": list(categories),
        },
    )


__all__ = ["check_harmful"]
