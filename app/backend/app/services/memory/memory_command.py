"""显式长期记忆命令解析，仅接受用户明确表达的记忆意图。"""

from __future__ import annotations

import re

from app.services.memory.memory_policy import PREFERENCE_RULES


_EXPLICIT_PREFIXES = ("请记住", "请记下", "以后默认", "今后默认", "今后都")
_PHRASES = {
    "response_language": (("用中文", "使用中文", "中文回答"), "zh-CN"),
    "response_detail_level": (("简短", "简洁"), "brief"),
    "preferred_output_format": (("表格",), "table"),
    "default_currency": (("人民币", "元人民币"), "CNY"),
    "default_market": (("A股", "中国股市"), "CN"),
    "default_compare_period": (("同比",), "YoY"),
}


def parse_memory_command(text: str) -> tuple[str, str] | None:
    """解析显式偏好命令，无法确定时返回 None。"""
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized or not normalized.startswith(_EXPLICIT_PREFIXES):
        return None
    for memory_key, (phrases, value) in _PHRASES.items():
        if any(phrase in normalized for phrase in phrases):
            return memory_key, value
    return None


def extract_memory_candidate(text: str) -> tuple[str, str] | None:
    """从普通表达提取候选，不识别显式命令，避免重复写入。"""
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized or normalized.startswith(_EXPLICIT_PREFIXES):
        return None
    candidate_phrases = {
        "response_language": (("偏好中文", "喜欢用中文", "习惯用中文"), "zh-CN"),
        "preferred_output_format": (("喜欢表格", "偏好表格", "习惯看表格"), "table"),
        "default_compare_period": (("习惯看同比", "比较习惯用同比"), "YoY"),
    }
    for key, (phrases, value) in candidate_phrases.items():
        if any(phrase in normalized for phrase in phrases):
            return key, value
    return None
