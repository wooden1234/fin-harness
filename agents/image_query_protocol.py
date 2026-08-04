"""多模态查询协议：用户意图与图像线索的结构化标记（单一来源）。

编排路由与 Vision 合成必须共用此处常量，禁止在各处复制提示词文案白名单。
"""

from __future__ import annotations

USER_INTENT_HEADER = "【用户意图】"
IMAGE_CLUE_HEADER = "[图像线索·未核验]"
# 兼容旧合成格式
IMAGE_CLUE_HEADER_LEGACY = "[图像理解·未核验]"
DEFAULT_IMAGE_USER_QUERY = "请根据图片分析"

_IMAGE_CLUE_PREFIXES = (
    IMAGE_CLUE_HEADER,
    IMAGE_CLUE_HEADER_LEGACY,
    "【图像线索",
    "[图像线索",
    "[图像理解",
)


def has_image_clue_block(query: str) -> bool:
    text = str(query or "")
    if USER_INTENT_HEADER in text:
        return True
    return any(marker in text for marker in _IMAGE_CLUE_PREFIXES)


def extract_user_intent(query: str) -> str:
    """从合成消息中取出用户意图；无协议头时返回原文。"""
    text = str(query or "").strip()
    if not text:
        return ""
    if USER_INTENT_HEADER in text:
        after = text.split(USER_INTENT_HEADER, 1)[1]
        for sep in (IMAGE_CLUE_HEADER, IMAGE_CLUE_HEADER_LEGACY, "【图像", "\n\n["):
            if sep in after:
                after = after.split(sep, 1)[0]
        return after.strip()
    for marker in (IMAGE_CLUE_HEADER, IMAGE_CLUE_HEADER_LEGACY):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text


def wrap_user_intent(user_text: str) -> str:
    text = str(user_text or "").strip() or DEFAULT_IMAGE_USER_QUERY
    return f"{USER_INTENT_HEADER}\n{text}"


def wrap_image_clue_failed() -> str:
    return (
        f"{IMAGE_CLUE_HEADER}\n"
        "图片未能解析。请仅依据用户意图回答，并说明图像信息不可用。"
    )


__all__ = [
    "DEFAULT_IMAGE_USER_QUERY",
    "IMAGE_CLUE_HEADER",
    "IMAGE_CLUE_HEADER_LEGACY",
    "USER_INTENT_HEADER",
    "extract_user_intent",
    "has_image_clue_block",
    "wrap_image_clue_failed",
    "wrap_user_intent",
]
