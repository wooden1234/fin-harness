"""输入敏感身份信息规则。"""

from __future__ import annotations

from datetime import date
import re
from typing import Callable

from agents.guardrails.contracts import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailFinding,
    GuardrailSeverity,
    GuardrailStage,
    allow_input,
)
from agents.guardrails.input.normalization import NormalizedInput, ensure_normalized

PII_PATTERNS = (
    (
        r"(?<!\d)(?:\d{15}|\d{17}[0-9Xx])(?!\d)",
        "身份证号",
        "pii.identity_number",
    ),
    (r"(?<!\d)(?:\d{16}|\d{19})(?!\d)", "银行卡号", "pii.bank_card"),
    (r"(?<!\d)1[3-9]\d{9}(?!\d)", "手机号", "pii.phone_number"),
)

_PII_SEVERITIES = {
    "pii.identity_number": GuardrailSeverity.HIGH,
    "pii.bank_card": GuardrailSeverity.HIGH,
    "pii.phone_number": GuardrailSeverity.MEDIUM,
}

_IDENTITY_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_IDENTITY_CHECK_CODES = "10X98765432"


def _has_valid_date(value: str, *, legacy: bool = False) -> bool:
    """校验身份证号编码中的出生日期。"""
    date_text = f"19{value[6:12]}" if legacy else value[6:14]
    try:
        parsed = date.fromisoformat(
            f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"
        )
    except ValueError:
        return False
    return parsed <= date.today()


def _is_valid_identity_number(value: str) -> bool:
    """校验大陆居民身份证日期与 18 位校验码。"""
    normalized = value.upper()
    if normalized[:6] == "000000":
        return False
    if len(normalized) == 15:
        return normalized.isdigit() and _has_valid_date(normalized, legacy=True)
    if len(normalized) != 18 or not normalized[:17].isdigit():
        return False
    if not _has_valid_date(normalized):
        return False
    checksum = sum(
        int(number) * weight
        for number, weight in zip(normalized[:17], _IDENTITY_WEIGHTS)
    )
    return normalized[-1] == _IDENTITY_CHECK_CODES[checksum % 11]


def _passes_luhn(value: str) -> bool:
    """使用 Luhn 算法排除格式相似的普通长数字。"""
    if not value.isdigit() or len(set(value)) == 1:
        return False
    total = 0
    parity = len(value) % 2
    for index, character in enumerate(value):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_valid_phone_number(value: str) -> bool:
    return bool(re.fullmatch(r"1[3-9]\d{9}", value))


_PII_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "pii.identity_number": _is_valid_identity_number,
    "pii.bank_card": _passes_luhn,
    "pii.phone_number": _is_valid_phone_number,
}


def _mask_pii(value: str, rule_id: str) -> str:
    """仅保留识别和业务核对所需的最少字符。"""
    if rule_id == "pii.phone_number":
        return f"{value[:3]}****{value[-4:]}"
    if rule_id == "pii.identity_number":
        return f"{value[:6]}********{value[-4:]}"
    return f"{value[:4]}{'*' * max(len(value) - 8, 4)}{value[-4:]}"


def check_pii(query: str | NormalizedInput) -> GuardrailDecision:
    """识别并脱敏允许继续处理的敏感身份信息。"""
    normalized = ensure_normalized(query)
    query_text = normalized.canonical
    safe_content = query_text
    findings: list[GuardrailFinding] = []
    matched_rules: list[str] = []
    highest_severity = GuardrailSeverity.NONE

    for pattern, pii_type, rule_id in PII_PATTERNS:
        validator = _PII_VALIDATORS[rule_id]
        matches = [
            match
            for match in re.finditer(pattern, query_text)
            if validator(match.group(0))
        ]
        if not matches:
            continue

        severity = _PII_SEVERITIES[rule_id]
        if severity == GuardrailSeverity.HIGH:
            highest_severity = GuardrailSeverity.HIGH
        elif highest_severity == GuardrailSeverity.NONE:
            highest_severity = severity
        matched_rules.append(rule_id)
        findings.append(
            GuardrailFinding(
                rule_id=rule_id,
                category="pii",
                severity=severity,
                message=f"检测到{pii_type}",
                spans=[match.span() for match in matches],
                metadata={
                    "pii_type": pii_type,
                    "count": len(matches),
                    "validated": True,
                    "span_source": "canonical",
                },
            )
        )
        safe_content = re.sub(
            pattern,
            lambda match, current_rule=rule_id, current_validator=validator: (
                _mask_pii(match.group(0), current_rule)
                if current_validator(match.group(0))
                else match.group(0)
            ),
            safe_content,
        )

    if not findings:
        return allow_input()

    return GuardrailDecision(
        action=GuardrailAction.REDACT,
        stage=GuardrailStage.INPUT,
        severity=highest_severity,
        reason_code="pii_redacted",
        reason="检测到敏感个人信息，已完成脱敏",
        safe_content=safe_content,
        findings=findings,
        matched_rules=matched_rules,
        metadata={"pii_types": [item.metadata["pii_type"] for item in findings]},
    )


__all__ = ["PII_PATTERNS", "check_pii"]
