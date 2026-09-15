"""Runtime-owned model request context helpers."""

from __future__ import annotations

from typing import Any

from harness.prompt.assembler import header_snapshot


def request_header(*, system: str, tools: list[dict[str, Any]], llm: Any) -> dict[str, Any]:
    """Build the durable request header recorded before model execution."""
    return header_snapshot(
        system=system,
        tools=tools,
        adapter_defaults=dict(getattr(llm, "adapter_defaults", {}) or {}),
    )
