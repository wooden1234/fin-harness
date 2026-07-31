"""不同 Agent 上下文空间共用的预算与压缩契约。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ContextSpaceType(str, Enum):
    """上下文空间类型；每个消息栈只能由一个空间管理。"""

    CONVERSATION = "conversation"
    TOOL_LOOP = "tool_loop"
    RESEARCH_RUN = "research_run"
    DEEP_AGENT_LOOP = "deep_agent_loop"


class ContextBudgetPolicy(BaseModel):
    """单个上下文空间的显式预算和防护次数。"""

    explicit_max_tokens: int = Field(gt=0)
    trigger_ratio: float = Field(default=0.75, gt=0, lt=1)
    target_ratio: float = Field(default=0.50, gt=0, lt=1)
    admission_ratio: float = Field(default=0.85, gt=0, le=1)
    approximate_safety_multiplier: float = Field(default=1.20, ge=1)
    max_compaction_rounds: int = Field(default=1, ge=0)
    max_summary_attempts_per_round: int = Field(default=2, ge=1)
    max_snips_per_round: int = Field(default=1, ge=0)
    max_provider_retries_per_invocation: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_ratios(self) -> "ContextBudgetPolicy":
        if not self.target_ratio < self.trigger_ratio < self.admission_ratio:
            raise ValueError("context_budget_ratios_must_be_target_trigger_admission")
        return self

    def trigger_tokens(self, effective_limit: int) -> int:
        return max(1, int(effective_limit * self.trigger_ratio))

    def target_tokens(self, effective_limit: int) -> int:
        return max(1, int(effective_limit * self.target_ratio))

    def admission_tokens(self, effective_limit: int) -> int:
        return max(1, int(effective_limit * self.admission_ratio))


class ContextCounters(BaseModel):
    """区分压缩轮、LLM 尝试、本地裁剪和 Provider 重试。"""

    compaction_round_count: int = Field(default=0, ge=0)
    summary_attempt_count: int = Field(default=0, ge=0)
    snip_count: int = Field(default=0, ge=0)
    provider_retry_count: int = Field(default=0, ge=0)


class ContextMeasurement(BaseModel):
    """一次模型调用前后的 Token 测量结果。"""

    estimated_tokens: int = Field(ge=0)
    actual_input_tokens: int | None = Field(default=None, ge=0)
    counter_kind: str
    model_max_tokens: int | None = Field(default=None, gt=0)
    effective_limit: int = Field(gt=0)
    utilization_ratio: float = Field(ge=0)
    trigger_exceeded: bool
    admission_exceeded: bool


class SnipMetadata(BaseModel):
    """最大块裁剪结果；不保存原始正文。"""

    changed: bool = False
    block_type: str = ""
    original_tokens: int = Field(default=0, ge=0)
    content_hash: str = ""
    message_ids: list[str] = Field(default_factory=list)


__all__ = [
    "ContextBudgetPolicy",
    "ContextCounters",
    "ContextMeasurement",
    "ContextSpaceType",
    "SnipMetadata",
]
