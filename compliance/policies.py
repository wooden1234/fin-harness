"""合规策略定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComplianceDecision:
    passed: bool
    reason: str = ""
    needs_human: bool = False
