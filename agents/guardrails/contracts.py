"""护栏模块共享的数据合同。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailAction(StrEnum):
    """护栏可返回的标准处置动作。"""

    ALLOW = "allow"
    REDACT = "redact"
    CLARIFY = "clarify"
    BLOCK = "block"
    ESCALATE = "escalate"


class GuardrailStage(StrEnum):
    """护栏决策发生的执行阶段。"""

    INPUT = "input"
    CONTEXT = "context"
    TOOL = "tool"
    OUTPUT = "output"


class GuardrailSeverity(StrEnum):
    """护栏风险严重程度。"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailFinding(BaseModel):
    """单条护栏规则产生的风险发现。"""

    rule_id: str
    category: str
    severity: GuardrailSeverity
    message: str = ""
    spans: list[tuple[int, int]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailDecision(BaseModel):
    """各护栏检查器统一返回的结构化决策。"""

    action: GuardrailAction
    stage: GuardrailStage
    severity: GuardrailSeverity = GuardrailSeverity.NONE
    reason_code: str = ""
    reason: str = ""
    safe_content: str | None = None
    findings: list[GuardrailFinding] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    policy_version: str = "input-v1"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        """放行原文或安全脱敏文本时允许继续执行。"""
        return self.action in {
            GuardrailAction.ALLOW,
            GuardrailAction.REDACT,
        }

    @property
    def passed(self) -> bool:
        """兼容旧布尔字段，其含义为当前请求是否可继续。"""
        return self.should_continue


def allow_input() -> GuardrailDecision:
    """创建输入阶段的默认放行决策。"""
    return GuardrailDecision(
        action=GuardrailAction.ALLOW,
        stage=GuardrailStage.INPUT,
    )


__all__ = [
    "GuardrailAction",
    "GuardrailDecision",
    "GuardrailFinding",
    "GuardrailSeverity",
    "GuardrailStage",
    "allow_input",
]
