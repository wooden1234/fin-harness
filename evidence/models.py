"""证据和结论模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Evidence:
    source: str
    snippet: str
    url: str = ""
    page: int | None = None
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Claim:
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0
