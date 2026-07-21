"""第一阶段近似 token 估算（生产可替换为真实 tokenizer）。

规则：
- 中日韩等宽字符：约 1 token / 2 字符
- 其余（英文、数字、符号）：约 1 token / 4 字符
"""

from __future__ import annotations

import re

from langchain_core.messages import AnyMessage

# 16K 输入预算（系统提示 + 摘要 + 问题 + 近期消息 + 余量）
CONTEXT_TOKEN_BUDGET = 16_000
COMPRESS_TRIGGER_TOKENS = 12_000
POST_COMPRESS_TOKENS = 8_000
SUMMARY_TOKEN_LIMIT = 1_200
# 单条消息上限：避免 SQL / 表格 / JSON 工具结果占满窗口
MAX_SINGLE_MESSAGE_TOKENS = 2_000

_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)


def estimate_tokens(text: str) -> int:
    """近似估算文本 token 数。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return (cjk + 1) // 2 + (other + 3) // 4


def message_text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def estimate_message_tokens(message: AnyMessage) -> int:
    """单条消息 token（已施加单条上限）。"""
    return min(estimate_tokens(message_text(message)), MAX_SINGLE_MESSAGE_TOKENS)


def truncate_to_token_limit(text: str, limit: int) -> str:
    """将文本截断到不超过 ``limit`` 近似 tokens。"""
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    clipped = text[:lo].rstrip()
    return f"{clipped}…" if clipped else text[:1]


def capped_message_text(message: AnyMessage) -> str:
    """用于摘要输入 / 展示的单条消息文本（带单条上限）。"""
    return truncate_to_token_limit(message_text(message), MAX_SINGLE_MESSAGE_TOKENS)
