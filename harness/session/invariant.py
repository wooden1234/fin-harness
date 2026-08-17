"""请求可从 log 重建。"""

from __future__ import annotations

from collections.abc import Sequence

from harness.contracts.errors import InvariantError
from harness.session.types import SessionEvent


def assert_model_request_logged(events: Sequence[SessionEvent], *, turn: int, step: int) -> None:
    has_header = any(
        event.event_type == "request/header" and event.turn == turn and event.step == step
        for event in events
    )
    if not has_header:
        raise InvariantError("缺少 request/header", code="missing_request_header")
