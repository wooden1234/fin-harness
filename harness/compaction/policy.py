"""压缩策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CompactPolicy:
    context_window: int = 65536
    trigger_ratio: float = 0.8
    retain_ratio: float = 0.16
    max_overflow_retries: int = 1
    prune_chars: int = 8192
    head_chars: int = 4096
    tail_chars: int = 1024
    max_summary_turns: int = 10
    max_narrative_chars: int = 400


def policy_from_settings() -> CompactPolicy:
    try:
        from app.core.config import settings

        window = int(getattr(settings, "COMPACTION_CONTEXT_WINDOW", 65536) or 65536)
    except Exception:  # noqa: BLE001
        window = 65536
    return CompactPolicy(context_window=window)


def compact_limit(policy: CompactPolicy) -> int:
    return math.floor(policy.context_window * policy.trigger_ratio)


def retain_limit(policy: CompactPolicy) -> int:
    return math.floor(policy.context_window * policy.retain_ratio)
