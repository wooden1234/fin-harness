"""单轮结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    session_id: str
    run_id: str
    finish_reason: str
    published_answer: str | None = None
    follow_ups: list[str] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    waiting_approval: bool = False
    approval_id: str | None = None
    call_id: str | None = None
    error: str | None = None
