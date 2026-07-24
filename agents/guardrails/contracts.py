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


class GuardrailDecision(BaseModel):
    """各护栏检查器统一返回的结构化决策。"""

    action: GuardrailAction
    stage: GuardrailStage
    reason_code: str = ""
    reason: str = ""
    safe_content: str | None = None
    matched_rules: list[str] = Field(default_factory=list)
    policy_version: str = "input-v1"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """当前链路仅放行原文；脱敏动作需先接入内容替换节点。"""
        return self.action == GuardrailAction.ALLOW


def allow_input() -> GuardrailDecision:
    """创建输入阶段的默认放行决策。"""
    return GuardrailDecision(
        action=GuardrailAction.ALLOW,
        stage=GuardrailStage.INPUT,
    )


__all__ = [
    "GuardrailAction",
    "GuardrailDecision",
    "GuardrailStage",
    "allow_input",
]
