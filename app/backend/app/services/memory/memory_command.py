"""长期偏好动作分类与确定性规则提取。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from app.services.memory.memory_policy import (
    contains_persistent_memory_marker,
    contains_temporary_memory_marker,
    find_sensitive_memory_rules,
)


MemoryActionKind = Literal[
    "remember",
    "update",
    "delete",
    "temporary",
    "sensitive",
    "implicit",
    "ordinary",
]


@dataclass(frozen=True, slots=True)
class MemoryRuleAction:
    kind: MemoryActionKind
    memory_key: str | None = None
    value: str | None = None

    @property
    def resolved(self) -> bool:
        if self.kind == "delete":
            return self.memory_key is not None
        if self.kind in {"remember", "update"}:
            return self.memory_key is not None and self.value is not None
        return True


_EXPLICIT_PREFIXES = ("请记住", "请记下", "以后默认", "今后默认", "今后都")
_UPDATE_MARKERS = ("改成", "改为", "修改成", "修改为", "以后改用", "今后改用")
_AMBIGUOUS_UPDATE_MARKERS = ("修改", "更改", "调整")
_DELETE_MARKERS = ("忘记", "删除", "清除", "不要再记住", "不再记住")
_MEMORY_MANAGEMENT_HINTS = ("记忆", "偏好", "默认", "之前那个", "刚才那个", "你记住")

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "response_language": ("回答语言", "语言", "中文", "英文", "英语"),
    "response_detail_level": ("详细程度", "回答长度", "回答详情", "简短", "详细"),
    "preferred_output_format": ("输出格式", "展示格式", "格式", "表格", "markdown", "纯文本"),
    "default_currency": ("默认币种", "币种", "货币", "人民币", "美元", "港币"),
    "default_market": ("默认市场", "市场", "a股", "港股", "美股"),
    "default_compare_period": ("比较周期", "对比周期", "对比方式", "同比", "环比"),
    "citation_preference": ("引用偏好", "引用来源", "引用", "来源"),
}

_VALUE_PHRASES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "response_language": (
        (("用中文", "使用中文", "中文回答", "中文"), "zh-CN"),
        (("用英文", "使用英文", "英文回答", "英语回答", "英文", "英语"), "en-US"),
    ),
    "response_detail_level": (
        (("简短", "简洁", "精简", "只看结论", "直接给结论"), "brief"),
        (("标准详细", "适中", "正常详细"), "standard"),
        (("详细", "详尽", "展开说明", "完整说明"), "detailed"),
    ),
    "preferred_output_format": (
        (("纯文本", "不用markdown", "不要markdown"), "plain_text"),
        (("markdown", "md格式"), "markdown"),
        (("表格",), "table"),
    ),
    "default_currency": (
        (("人民币", "元人民币", "cny", "rmb"), "CNY"),
        (("美元", "usd"), "USD"),
        (("港币", "hkd"), "HKD"),
    ),
    "default_market": (
        (("a股", "中国股市", "沪深市场", "cn市场"), "CN"),
        (("港股", "香港股市", "hk市场"), "HK"),
        (("美股", "美国股市", "us市场"), "US"),
    ),
    "default_compare_period": (
        (("同比", "yoy"), "YoY"),
        (("季度环比", "季环比", "qoq"), "QoQ"),
        (("月度环比", "月环比", "mom"), "MoM"),
    ),
    "citation_preference": (
        (("总是引用", "始终引用", "每次引用", "必须引用"), "always"),
        (("有来源时引用", "可用时引用", "有依据时引用"), "when_available"),
        (("不要引用", "无需引用", "不需要引用"), "never"),
    ),
}

_IMPLICIT_PHRASES = {
    "response_language": (("偏好中文", "喜欢用中文", "习惯用中文"), "zh-CN"),
    "preferred_output_format": (("喜欢表格", "偏好表格", "习惯看表格"), "table"),
    "default_compare_period": (("习惯看同比", "比较习惯用同比"), "YoY"),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _extract_value(normalized: str) -> tuple[str, str] | None:
    for memory_key, candidates in _VALUE_PHRASES.items():
        for phrases, value in candidates:
            if any(phrase in normalized for phrase in phrases):
                return memory_key, value
    return None


def extract_turn_preferences(text: str) -> dict[str, str]:
    """提取只在当前轮生效的确定性偏好，不写入长期记忆。"""
    if not contains_temporary_memory_marker(text):
        return {}
    normalized = _normalize(text)
    preferences: dict[str, str] = {}
    for memory_key, candidates in _VALUE_PHRASES.items():
        for phrases, value in candidates:
            if any(phrase in normalized for phrase in phrases):
                preferences[memory_key] = value
                break
    return preferences


def _extract_key(normalized: str) -> str | None:
    for memory_key, aliases in _KEY_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return memory_key
    return None


def _extract_update_key(normalized: str) -> str | None:
    positions = [
        normalized.find(marker)
        for marker in _UPDATE_MARKERS
        if marker in normalized
    ]
    if not positions:
        return None
    return _extract_key(normalized[:min(positions)])


def parse_memory_rule_action(text: str) -> MemoryRuleAction:
    """仅用本地规则分类动作；同步链路不得调用大模型补全。"""
    normalized = _normalize(text)
    if find_sensitive_memory_rules(text):
        return MemoryRuleAction("sensitive")
    if contains_temporary_memory_marker(text):
        return MemoryRuleAction("temporary")

    if any(marker in normalized for marker in _DELETE_MARKERS):
        key = _extract_key(normalized)
        if key is not None or any(
            hint in normalized for hint in _MEMORY_MANAGEMENT_HINTS
        ):
            return MemoryRuleAction("delete", memory_key=key)

    extracted = _extract_value(normalized)
    if any(marker in normalized for marker in _UPDATE_MARKERS):
        key = _extract_update_key(normalized)
        if extracted is not None:
            inferred_key, value = extracted
            if key is not None and key != inferred_key:
                return MemoryRuleAction("update", memory_key=key)
            return MemoryRuleAction("update", memory_key=key or inferred_key, value=value)
        return MemoryRuleAction("update", memory_key=key)

    if (
        any(marker in normalized for marker in _AMBIGUOUS_UPDATE_MARKERS)
        and any(hint in normalized for hint in _MEMORY_MANAGEMENT_HINTS)
    ):
        return MemoryRuleAction("update", memory_key=_extract_key(normalized))

    if normalized.startswith(_EXPLICIT_PREFIXES):
        if extracted is None:
            return MemoryRuleAction("remember")
        memory_key, value = extracted
        return MemoryRuleAction("remember", memory_key=memory_key, value=value)

    if contains_persistent_memory_marker(text):
        return MemoryRuleAction("implicit")
    return MemoryRuleAction("ordinary")


def parse_memory_command(text: str) -> tuple[str, str] | None:
    """兼容显式新增偏好的旧接口，无法确定时返回 None。"""
    action = parse_memory_rule_action(text)
    if action.kind != "remember" or not action.resolved:
        return None
    return action.memory_key, action.value


def extract_preference_rule(text: str) -> tuple[str, str] | None:
    """从普通表达提取长期偏好，不重复处理同步动作。"""
    if parse_memory_rule_action(text).kind != "implicit":
        return None
    normalized = _normalize(text)
    for key, (phrases, value) in _IMPLICIT_PHRASES.items():
        if any(phrase in normalized for phrase in phrases):
            return key, value
    return None


__all__ = [
    "MemoryActionKind",
    "MemoryRuleAction",
    "extract_turn_preferences",
    "extract_preference_rule",
    "parse_memory_command",
    "parse_memory_rule_action",
]
