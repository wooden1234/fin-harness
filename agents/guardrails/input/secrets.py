"""输入中的凭据与秘密检测。"""

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


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "secret.private_key",
        re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----", re.IGNORECASE),
    ),
    (
        "secret.jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "secret.api_token",
        re.compile(
            r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|"
            r"gh[pousr]_[A-Za-z0-9_]{16,})"
            r"(?![A-Za-z0-9_-])",
            re.IGNORECASE,
        ),
    ),
    (
        "secret.labeled_value",
        re.compile(
            r"(?:密码|口令|验证码|私钥|password|passwd|pwd|"
            r"api[\s_-]?key|access[\s_-]?token|token|secret)"
            r"\s*(?:是|为|[:=])\s*([^\s,，;；]{4,})",
            re.IGNORECASE,
        ),
    ),
)

_QUESTION_PREFIXES = ("如何", "怎么", "怎样", "什么", "哪里", "是否")


def check_secrets(query: str | NormalizedInput) -> GuardrailDecision:
    """阻断包含实际凭据值的输入，避免进入持久化和记忆链路。"""
    normalized = ensure_normalized(query)
    findings: list[GuardrailFinding] = []
    for rule_id, pattern in _SECRET_PATTERNS:
        spans: list[tuple[int, int]] = []
        for match in pattern.finditer(normalized.canonical):
            captured = match.group(1) if match.lastindex else ""
            if captured and captured.startswith(_QUESTION_PREFIXES):
                continue
            spans.append(match.span())
        if spans:
            findings.append(
                GuardrailFinding(
                    rule_id=rule_id,
                    category="secret",
                    severity=GuardrailSeverity.CRITICAL,
                    message="检测到凭据或秘密值",
                    spans=spans,
                )
            )
    if not findings:
        return allow_input()
    return GuardrailDecision(
        action=GuardrailAction.BLOCK,
        stage=GuardrailStage.INPUT,
        severity=GuardrailSeverity.CRITICAL,
        reason_code="secret_detected",
        reason="输入中包含密码、令牌、验证码或私钥等秘密信息",
        findings=findings,
        matched_rules=[finding.rule_id for finding in findings],
    )


__all__ = ["check_secrets"]
