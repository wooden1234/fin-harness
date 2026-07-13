"""一次 Agent 运行的上下文定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class RunContext:
    """贯穿一次请求的运行上下文。"""

    user_id: str | None = None
    conversation_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    roles: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def build_run_context(
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
    roles: tuple[str, ...] = (),
    permissions: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        user_id=user_id,
        conversation_id=conversation_id,
        roles=roles,
        permissions=permissions,
        metadata=metadata or {},
    )
