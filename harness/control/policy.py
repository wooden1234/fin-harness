"""Per-turn control policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.finalization.submit import finalize_markdown
from harness.tools.error_policy import decide_after_tools


class TurnPolicy:
    """Centralizes product decisions around tool results and publication."""

    def decide_after_tools(self, results: Sequence[Any], *, events, turn: int):
        return decide_after_tools(results, events=events, turn=turn)

    def finalize(self, markdown: str) -> str:
        return finalize_markdown(markdown)
