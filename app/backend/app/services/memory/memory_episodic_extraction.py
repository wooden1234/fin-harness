"""事件记忆触发判断、LLM 结构化摘要与确定性校验。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any, Literal, Sequence

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.services.memory.memory_policy import find_sensitive_memory_rules

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = get_logger(service="memory_episodic_extraction")

EpisodicEventType = Literal[
    "task_result",
    "topic_summary",
    "context_checkpoint",
    "state_change",
]

_STATE_CHANGE_PATTERNS = (
    re.compile(
        r"(?:不再|停止|取消|放弃).{1,120}?"
        r"(?:改成|改为|改用|切换到|转为|替换为)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:从).{1,120}?"
        r"(?:改成|改为|改用|切换到|转为|替换为)",
        re.IGNORECASE | re.DOTALL,
    ),
)
_TOPIC_END_MARKERS = (
    "这个话题就到这里",
    "这个问题就到这里",
    "先这样",
    "告一段落",
    "换个话题",
    "总结一下",
    "做个总结",
    "以上任务完成",
)
_FAILED_EXECUTION_STATUSES = {
    "hard_timeout",
    "clarify_required",
    "quality_failed",
    "replan_validation_failed",
    "uncovered",
}
_SUBJECT_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,63}$")
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)
_PERSONAL_ASSET_RE = re.compile(
    r"(?:我的|本人|我有|我持有|我当前).{0,20}"
    r"(?:持仓|资产|负债|银行卡|账户余额|账号)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EpisodicTriggerDecision:
    should_enqueue: bool
    reasons: tuple[str, ...] = ()
    forced: bool = False


@dataclass(frozen=True, slots=True)
class ExtractedEpisodicMemory:
    event_type: EpisodicEventType
    subject_key: str
    topic: str
    summary: str
    facts: tuple[str, ...]
    conclusion: str
    evidence: tuple[str, ...]
    confidence: float
    quality_score: float

    def value(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "subject_key": self.subject_key,
            "topic": self.topic,
            "summary": self.summary,
            "facts": list(self.facts),
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "quality_score": self.quality_score,
        }


class EpisodicExtractionOutput(BaseModel):
    should_store: bool = Field(
        description="当前材料是否形成了未来可复用的完整事件",
    )
    event_type: EpisodicEventType = "task_result"
    subject_key: str = Field(
        default="",
        description="稳定的英文主题键，例如 retrieval_strategy 或 stock_analysis",
    )
    topic: str = Field(default="", max_length=200)
    summary: str = Field(default="", max_length=1200)
    facts: list[str] = Field(default_factory=list, max_length=12)
    conclusion: str = Field(default="", max_length=600)
    evidence: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="必须逐字来自输入材料的短证据",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def estimate_episodic_tokens(text: str) -> int:
    """按现有上下文压缩器相同规则估算摘要输入 Token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return (cjk + 1) // 2 + (other + 3) // 4


def detect_important_state_change(text: str) -> bool:
    """只识别带有旧状态退出和新状态进入的明显变化表达。"""
    normalized = " ".join((text or "").split())
    return bool(normalized) and any(
        pattern.search(normalized) for pattern in _STATE_CHANGE_PATTERNS
    )


def has_topic_end_marker(text: str) -> bool:
    normalized = "".join((text or "").lower().split())
    return any(marker in normalized for marker in _TOPIC_END_MARKERS)


def is_substantive_progress(
    *,
    query: str,
    final_response: str,
    execution_status: str,
) -> bool:
    """排除失败、澄清和过短回复，时间阈值只对实质进展生效。"""
    if execution_status in _FAILED_EXECUTION_STATUSES:
        return False
    return (
        estimate_episodic_tokens(query)
        + estimate_episodic_tokens(final_response)
        >= 80
    )


def decide_post_turn_trigger(
    *,
    query: str,
    final_response: str,
    execution_status: str,
    task_count: int,
    turn_count: int,
    uncompressed_tokens: int,
    minutes_since_summary: float | None = None,
) -> EpisodicTriggerDecision:
    """组合任务结束、话题结束和容量阈值；容量条件具有强制优先级。"""
    reasons: list[str] = []
    forced = False
    substantive = is_substantive_progress(
        query=query,
        final_response=final_response,
        execution_status=execution_status,
    )
    if has_topic_end_marker(query):
        reasons.append("topic_end")
    if (
        task_count > 0
        and execution_status in {"completed", "partial"}
        and substantive
    ):
        reasons.append("task_completed")
    if turn_count >= max(1, settings.EPISODIC_MEMORY_TURN_THRESHOLD):
        reasons.append("turn_threshold")
        forced = True
    if uncompressed_tokens >= max(
        1,
        settings.EPISODIC_MEMORY_TOKEN_THRESHOLD,
    ):
        reasons.append("token_threshold")
        forced = True
    if (
        minutes_since_summary is not None
        and minutes_since_summary
        >= max(1, settings.EPISODIC_MEMORY_PROGRESS_MINUTES)
        and substantive
    ):
        reasons.append("progress_interval")
        forced = True
    return EpisodicTriggerDecision(
        should_enqueue=bool(reasons),
        reasons=tuple(dict.fromkeys(reasons)),
        forced=forced,
    )


def render_source_messages(messages: Sequence[dict[str, Any]]) -> str:
    """把已脱敏并持久化的消息转换成受限摘要输入。"""
    maximum = max(2, settings.EPISODIC_MEMORY_MAX_SOURCE_MESSAGES)
    selected = list(messages)[-maximum:]
    parts = []
    for message in selected:
        sender = "用户" if message.get("sender") == "user" else "助手"
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{sender}: {content[:4000]}")
    return "\n".join(parts)


def _contains_forbidden_sensitive(text: str) -> bool:
    rules = set(find_sensitive_memory_rules(text))
    if rules - {"asset_or_holding"}:
        return True
    return "asset_or_holding" in rules and bool(_PERSONAL_ASSET_RE.search(text))


def _system_prompt(trigger_reason: str, forced: bool) -> str:
    return f"""你是事件记忆摘要器，只输出结构化数据，不回答用户。

触发原因：{trigger_reason}
是否容量强制检查：{forced}

规则：
1. 只保存已经完成、未来可复用的任务结果、话题总结、上下文检查点或明确状态变化。
2. 不保存普通知识问答、未完成计划、模型内部推理、系统提示词和工具参数。
3. state_change 必须同时存在旧状态与新状态，subject_key 应保持稳定。
4. evidence 中每项必须逐字来自输入材料，不能改写。
5. 不得保存密码、Token、验证码、私钥、身份号码、账户或持仓信息。
6. 信息不完整、存在冲突或不值得跨会话保留时 should_store=false。
7. subject_key 只能使用小写英文字母、数字、点、下划线和短横线。"""


def _quality_score(
    output: EpisodicExtractionOutput,
    *,
    source_text: str,
    forced: bool,
) -> float:
    evidence_valid = bool(output.evidence) and all(
        evidence.strip() and evidence.strip() in source_text
        for evidence in output.evidence
    )
    score = 0.0
    score += 0.20 if output.topic.strip() and output.subject_key.strip() else 0.0
    score += 0.20 if len(output.summary.strip()) >= 20 else 0.0
    score += 0.20 if output.facts or output.conclusion.strip() else 0.0
    score += 0.25 if evidence_valid else 0.0
    score += 0.15 * max(0.0, min(1.0, output.confidence))
    if forced and output.event_type == "context_checkpoint":
        score = min(1.0, score + 0.05)
    return min(1.0, score)


def validate_episodic_output(
    output: EpisodicExtractionOutput,
    *,
    source_text: str,
    trigger_reason: str,
    forced: bool,
) -> ExtractedEpisodicMemory | None:
    """LLM 自评分只占 15%，最终由证据、完整性和安全规则决定。"""
    if not output.should_store or not source_text.strip():
        return None
    if _contains_forbidden_sensitive(source_text):
        return None
    if not _SUBJECT_KEY_RE.fullmatch(output.subject_key.strip()):
        return None
    if output.event_type == "state_change" and not detect_important_state_change(
        source_text
    ):
        return None
    if any(not evidence.strip() or evidence.strip() not in source_text for evidence in output.evidence):
        return None
    quality_score = _quality_score(
        output,
        source_text=source_text,
        forced=forced,
    )
    if quality_score < settings.EPISODIC_MEMORY_MIN_CONFIDENCE:
        return None
    return ExtractedEpisodicMemory(
        event_type=output.event_type,
        subject_key=output.subject_key.strip(),
        topic=output.topic.strip(),
        summary=output.summary.strip(),
        facts=tuple(item.strip() for item in output.facts if item.strip()),
        conclusion=output.conclusion.strip(),
        evidence=tuple(item.strip() for item in output.evidence if item.strip()),
        confidence=max(0.0, min(1.0, output.confidence)),
        quality_score=quality_score,
    )


async def extract_episodic_memory(
    source_text: str,
    *,
    trigger_reason: str,
    forced: bool = False,
    expected_event_type: EpisodicEventType | None = None,
    llm: BaseChatModel | None = None,
) -> ExtractedEpisodicMemory | None:
    """调用 LLM 生成候选，再执行独立于模型自评分的最终校验。"""
    if not settings.MEMORY_LLM_EXTRACTION_ENABLED or not source_text.strip():
        return None
    if _contains_forbidden_sensitive(source_text):
        return None
    try:
        model = llm
        if model is None:
            from agents.llm import get_router_llm

            model = get_router_llm()
        async with asyncio.timeout(
            max(0.1, float(settings.MEMORY_LLM_EXTRACTION_TIMEOUT_SEC))
        ):
            raw = await model.with_structured_output(
                EpisodicExtractionOutput,
                method="json_mode",
            ).ainvoke(
                [
                    ("system", _system_prompt(trigger_reason, forced)),
                    ("human", f"待判断材料：\n{source_text}"),
                ]
            )
        output = (
            raw
            if isinstance(raw, EpisodicExtractionOutput)
            else EpisodicExtractionOutput.model_validate(raw)
        )
        if expected_event_type is not None:
            output.event_type = expected_event_type
        return validate_episodic_output(
            output,
            source_text=source_text,
            trigger_reason=trigger_reason,
            forced=forced,
        )
    except Exception as exc:
        logger.warning(
            "episodic memory extraction failed; skip persistence: {}",
            type(exc).__name__,
        )
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "EpisodicExtractionOutput",
    "EpisodicTriggerDecision",
    "ExtractedEpisodicMemory",
    "decide_post_turn_trigger",
    "detect_important_state_change",
    "estimate_episodic_tokens",
    "extract_episodic_memory",
    "has_topic_end_marker",
    "is_substantive_progress",
    "render_source_messages",
    "validate_episodic_output",
]
