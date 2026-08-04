"""聊天图片 Vision 预处理（Qwen 多模态，仅看图一步）。

模型实例统一从 agents.llm.get_vision_llm 获取，禁止在此直连 API。
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from agents.image_query_protocol import (
    IMAGE_CLUE_HEADER,
    USER_INTENT_HEADER,
    wrap_image_clue_failed,
    wrap_user_intent,
)
from agents.llm import get_vision_llm
from app.core.logger import get_logger

logger = get_logger(service="vision")

_VISION_SYSTEM = """你是金融场景的图像理解助手。根据用户图片输出简洁 JSON，字段：
image_type (string), summary (string), entities (string[]), metrics (string[]),
uncertainties (string[]), suggested_focus (string)。
规则：
- summary 用 1-3 句中文概括图中可见信息，不要重复整段 JSON；
- metrics 列出图中可见数值（标明未核验），每项一句短文本；
- suggested_focus 用一句中文说明最值得结合用户问题关注的点；
- 读不清的写入 uncertainties；
- 不要编造图中没有的信息；不要给出买卖建议。
只输出一个 JSON 对象，不要 Markdown 代码块。"""


class ImageUnderstanding(BaseModel):
    image_type: str = Field(default="unknown", max_length=64)
    summary: str = Field(default="", max_length=2000)
    entities: list[str] = Field(default_factory=list, max_length=32)
    metrics: list[str] = Field(default_factory=list, max_length=32)
    uncertainties: list[str] = Field(default_factory=list, max_length=16)
    suggested_focus: str = Field(default="", max_length=500)

    @field_validator("entities", "metrics", "uncertainties", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        return [str(value).strip()] if str(value).strip() else []

    @field_validator("suggested_focus", mode="before")
    @classmethod
    def _coerce_focus(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return "；".join(parts[:6])
        return str(value).strip()


def render_image_understanding_block(understanding: ImageUnderstanding) -> str:
    """紧凑结构化线索，避免淹没用户意图。"""
    lines = [
        IMAGE_CLUE_HEADER,
        f"- 类型：{understanding.image_type or 'unknown'}",
        f"- 摘要：{(understanding.summary or '（无）').strip()}",
    ]
    if understanding.entities:
        lines.append("- 实体：" + "、".join(understanding.entities[:12]))
    if understanding.metrics:
        lines.append("- 可见数值：" + "；".join(understanding.metrics[:10]))
    if understanding.uncertainties:
        lines.append("- 不确定：" + "；".join(understanding.uncertainties[:6]))
    if understanding.suggested_focus:
        lines.append(f"- 可结合关注：{understanding.suggested_focus}")
    lines.append(
        f"- 说明：图像数值未核验；回答必须紧扣{USER_INTENT_HEADER}，图像仅作线索。"
    )
    return "\n".join(lines)


def compose_effective_query(user_text: str, understanding: ImageUnderstanding | None) -> str:
    """用户意图置顶，图像理解降为结构化附属线索。"""
    intent_block = wrap_user_intent(user_text)
    if understanding is None:
        return f"{intent_block}\n\n{wrap_image_clue_failed()}"
    return f"{intent_block}\n\n{render_image_understanding_block(understanding)}"


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_understanding(raw: str) -> ImageUnderstanding:
    data = _extract_json_object(raw)
    if data is not None:
        try:
            return ImageUnderstanding.model_validate(data)
        except Exception:
            logger.warning("vision json schema coerce failed, using partial fields")
            return ImageUnderstanding(
                image_type=str(data.get("image_type") or "unknown")[:64],
                summary=str(data.get("summary") or "")[:2000],
                entities=_coerce_list_field(data.get("entities")),
                metrics=_coerce_list_field(data.get("metrics")),
                uncertainties=_coerce_list_field(data.get("uncertainties")),
                suggested_focus=_coerce_focus_field(data.get("suggested_focus")),
            )
    # 非 JSON：只保留短摘要，避免把噪声整段灌进下游
    fallback = str(raw or "").strip()
    if fallback.startswith("{") or '"image_type"' in fallback:
        fallback = "模型返回未能结构化解析"
    return ImageUnderstanding(
        image_type="unknown",
        summary=fallback[:500] or "未能结构化解析图片",
    )


def _coerce_list_field(value: Any) -> list[str]:
    return ImageUnderstanding._coerce_str_list(value)


def _coerce_focus_field(value: Any) -> str:
    return ImageUnderstanding._coerce_focus(value)


def _to_data_url(*, image_bytes: bytes, content_type: str) -> str:
    mime = (content_type or "image/png").split(";")[0].strip() or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        ]
        return "\n".join(item for item in texts if item)
    return str(content or "")


async def understand_image(
    *,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
    content_type: str = "image/png",
    user_text: str = "",
) -> ImageUnderstanding:
    # 优先 base64：由本服务从 OSS 拉图，避免 DashScope 再去拉签名 URL 超时
    if image_bytes:
        media_url = _to_data_url(image_bytes=image_bytes, content_type=content_type)
    elif image_url:
        media_url = image_url
    else:
        raise ValueError("image_bytes 与 image_url 不能同时为空")

    prompt = (
        "请理解这张图片，并按系统要求输出 JSON。"
        + (
            f"\n用户意图（回答时需服务该意图，勿被图像细节带偏）：{user_text.strip()}"
            if user_text.strip()
            else ""
        )
    )
    messages = [
        SystemMessage(content=_VISION_SYSTEM),
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": media_url}},
                {"type": "text", "text": prompt},
            ]
        ),
    ]
    try:
        response = await get_vision_llm().ainvoke(messages)
    except Exception:
        logger.exception("vision llm invoke failed")
        raise
    return _parse_understanding(
        _message_content_to_text(getattr(response, "content", ""))
    )


__all__ = [
    "ImageUnderstanding",
    "compose_effective_query",
    "render_image_understanding_block",
    "understand_image",
]
