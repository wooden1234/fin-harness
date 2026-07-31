"""输入护栏使用的确定性文本规范化。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class NormalizedInput:
    """保留原文，并提供检测用的规范文本和紧凑文本。"""

    original: str
    canonical: str
    compact: str


def normalize_input(text: str) -> NormalizedInput:
    """统一 Unicode、不可见字符和空白，降低简单混淆绕过概率。"""
    normalized = unicodedata.normalize("NFKC", text)
    visible_chars: list[str] = []
    for char in normalized:
        if char.isspace():
            visible_chars.append(" ")
            continue
        if unicodedata.category(char) in {"Cc", "Cf"}:
            continue
        visible_chars.append(char)

    canonical = re.sub(r"\s+", " ", "".join(visible_chars)).strip()
    compact = re.sub(r"\s+", "", canonical)
    return NormalizedInput(
        original=text,
        canonical=canonical,
        compact=compact,
    )


def ensure_normalized(value: str | NormalizedInput) -> NormalizedInput:
    """兼容公共检查器直接接收字符串的调用方式。"""
    if isinstance(value, NormalizedInput):
        return value
    return normalize_input(value)


__all__ = ["NormalizedInput", "ensure_normalized", "normalize_input"]
