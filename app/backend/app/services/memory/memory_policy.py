"""长期记忆偏好白名单与值校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreferenceRule:
    value_type: str
    choices: tuple[str, ...] = ()
    ttl_days: int | None = None


PREFERENCE_RULES: dict[str, PreferenceRule] = {
    "response_language": PreferenceRule("enum", ("zh-CN", "en-US")),
    "response_detail_level": PreferenceRule("enum", ("brief", "standard", "detailed"), 730),
    "preferred_output_format": PreferenceRule("enum", ("plain_text", "markdown", "table"), 365),
    "default_currency": PreferenceRule("enum", ("CNY", "USD", "HKD"), 365),
    "default_market": PreferenceRule("enum", ("CN", "HK", "US"), 180),
    "default_compare_period": PreferenceRule("enum", ("YoY", "QoQ", "MoM"), 180),
    "citation_preference": PreferenceRule("enum", ("always", "when_available", "never"), 365),
}


def validate_preference(memory_key: str, value: Any) -> PreferenceRule:
    rule = PREFERENCE_RULES.get(memory_key)
    if rule is None:
        raise ValueError("不支持的偏好 key")
    if rule.value_type == "enum" and (
        not isinstance(value, str) or value not in rule.choices
    ):
        raise ValueError(f"{memory_key} 的值必须是: {', '.join(rule.choices)}")
    return rule
