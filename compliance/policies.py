"""合规策略定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ComplianceAction = Literal["pass", "rewrite", "block", "escalate"]


@dataclass(frozen=True, slots=True)
class ComplianceDecision:
    action: ComplianceAction
    reason_code: str = ""
    reason: str = ""
    safe_answer: str = ""

    @property
    def passed(self) -> bool:
        """仅放行原文或已经生成确定性安全文案的结果。"""
        return self.action in {"pass", "rewrite"}
