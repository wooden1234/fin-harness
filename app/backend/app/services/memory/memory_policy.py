"""长期记忆偏好白名单与值校验。"""

from __future__ import annotations

import re
from typing import Any
import unicodedata

from app.services.memory.memory_catalog import (
    MemoryKeyDefinition,
    preference_definitions,
)


PreferenceRule = MemoryKeyDefinition
PREFERENCE_RULES: dict[str, PreferenceRule] = preference_definitions()

_PERSISTENT_MARKERS = (
    "请记住",
    "请记下",
    "以后",
    "今后",
    "往后",
    "长期",
    "一直",
    "默认",
    "每次",
    "都用",
    "总是",
    "始终",
    "习惯",
    "偏好",
    "喜欢",
    "fromnowon",
    "bydefault",
    "always",
    "prefer",
)

_TEMPORARY_MARKERS = (
    "这次",
    "本次",
    "这一轮",
    "当前问题",
    "今天",
    "现在",
    "暂时",
    "临时",
    "仅本次",
    "只在这次",
    "这个问题",
    "thistime",
    "today",
    "currently",
    "temporarily",
)

_SENSITIVE_RULES = (
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    (
        "asset_or_holding",
        re.compile(
            r"持仓|资产|负债|银行卡|账户余额|账号|身份证|手机号|手机号码|电话号码"
        ),
    ),
    (
        "credential_or_secret",
        re.compile(
            r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----|"
            r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|"
            r"gh[pousr]_[A-Za-z0-9_]{16,})(?![A-Za-z0-9_-])|"
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])|"
            r"(?:密码|口令|验证码|私钥|password|passwd|pwd|"
            r"api[\s_-]?key|access[\s_-]?token|token|secret)"
            r"\s*(?:是|为|[:=])\s*\S{4,}",
            re.IGNORECASE,
        ),
    ),
)

_PREFERENCE_EVIDENCE_HINTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("response_language", "zh-CN"): ("中文", "汉语", "zh-cn"),
    ("response_language", "en-US"): ("英文", "英语", "en-us"),
    (
        "response_detail_level",
        "brief",
    ): ("简短", "简洁", "精简", "简要", "直接给结论", "只看结论"),
    (
        "response_detail_level",
        "standard",
    ): ("标准详细", "适中", "正常详细", "一般详细"),
    (
        "response_detail_level",
        "detailed",
    ): ("详细", "详尽", "展开说明", "完整说明", "更多细节"),
    (
        "preferred_output_format",
        "plain_text",
    ): ("纯文本", "plaintext", "plain_text", "不用markdown", "不要markdown"),
    ("preferred_output_format", "markdown"): ("markdown", "md格式"),
    ("preferred_output_format", "table"): ("表格", "table"),
    ("default_currency", "CNY"): ("人民币", "元人民币", "cny", "rmb"),
    ("default_currency", "USD"): ("美元", "usd"),
    ("default_currency", "HKD"): ("港币", "hkd"),
    ("default_market", "CN"): ("a股", "中国股市", "沪深市场", "cn市场"),
    ("default_market", "HK"): ("港股", "香港股市", "hk市场"),
    ("default_market", "US"): ("美股", "美国股市", "us市场"),
    ("default_compare_period", "YoY"): ("同比", "yoy"),
    ("default_compare_period", "QoQ"): ("季度环比", "季环比", "qoq"),
    ("default_compare_period", "MoM"): ("月度环比", "月环比", "mom"),
    (
        "citation_preference",
        "always",
    ): ("总是引用", "始终引用", "每次引用", "必须引用", "alwayscite"),
    (
        "citation_preference",
        "when_available",
    ): ("有来源时引用", "可用时引用", "有依据时引用", "whenavailable"),
    (
        "citation_preference",
        "never",
    ): ("不要引用", "无需引用", "不需要引用", "nevercite"),
}


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"\s+", "", normalized)


def find_sensitive_memory_rules(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text or "")
    return tuple(
        name
        for name, pattern in _SENSITIVE_RULES
        if pattern.search(normalized)
    )


def contains_temporary_memory_marker(text: str) -> bool:
    """判断原文是否明确限定为本轮或短期要求。"""
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in _TEMPORARY_MARKERS)


def contains_persistent_memory_marker(text: str) -> bool:
    """判断原文是否包含可跨会话生效的明确语言信号。"""
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in _PERSISTENT_MARKERS)


def validate_preference(memory_key: str, value: Any) -> PreferenceRule:
    rule = PREFERENCE_RULES.get(memory_key)
    if rule is None:
        raise ValueError("不支持的偏好 key")
    if rule.value_type == "enum" and (
        not isinstance(value, str) or value not in rule.choices
    ):
        raise ValueError(f"{memory_key} 的值必须是: {', '.join(rule.choices)}")
    return rule


def validate_preference_evidence(
    *,
    text: str,
    memory_key: str,
    value: str,
    evidence: str,
) -> PreferenceRule:
    """校验提取证据确实支持可跨会话保存的非敏感偏好。"""
    rule = validate_preference(memory_key, value)
    normalized_text = _normalize_text(text)
    normalized_evidence = _normalize_text(evidence)

    if not normalized_evidence:
        raise ValueError("偏好证据不能为空")
    if normalized_evidence not in normalized_text:
        raise ValueError("偏好证据必须来自用户原文")
    if find_sensitive_memory_rules(text):
        raise ValueError("用户原文包含不允许写入长期记忆的敏感信息")
    if any(marker in normalized_evidence for marker in _TEMPORARY_MARKERS):
        raise ValueError("临时要求不能写入长期记忆")
    if not any(marker in normalized_evidence for marker in _PERSISTENT_MARKERS):
        raise ValueError("缺少可跨会话生效的长期偏好信号")

    hints = _PREFERENCE_EVIDENCE_HINTS.get((memory_key, value), ())
    if not hints or not any(
        _normalize_text(hint) in normalized_evidence
        for hint in hints
    ):
        raise ValueError("偏好证据与 key/value 不匹配")
    return rule


__all__ = [
    "PREFERENCE_RULES",
    "PreferenceRule",
    "contains_persistent_memory_marker",
    "contains_temporary_memory_marker",
    "find_sensitive_memory_rules",
    "validate_preference",
    "validate_preference_evidence",
]
