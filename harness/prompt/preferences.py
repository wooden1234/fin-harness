"""每步加载长期偏好投影与本轮临时覆盖；失败空结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PreferenceContext:
    preferences: dict[str, Any] = field(default_factory=dict)
    turn_overrides: dict[str, str] = field(default_factory=dict)


def _turn_overrides(events: Sequence[Any], turn: int) -> dict[str, str]:
    try:
        from app.services.memory.memory_command import extract_turn_preferences
    except Exception:  # noqa: BLE001
        return {}
    merged: dict[str, str] = {}
    for event in events:
        if getattr(event, "turn", None) != turn:
            continue
        if getattr(event, "event_type", None) != "user/message":
            continue
        data = getattr(event, "data", None) or {}
        if not isinstance(data, Mapping):
            continue
        if str(data.get("source") or "user") != "user":
            continue
        merged.update(extract_turn_preferences(str(data.get("content") or "")))
    return merged


async def load_preference_context(
    *,
    store: Any,
    session_id: str,
    events: Sequence[Any],
    turn: int,
) -> PreferenceContext:
    """从 session header 加载 fin_agent 白名单偏好；任何失败都返回空偏好。"""
    turn_overrides = _turn_overrides(events, turn)
    try:
        header = await store.get(session_id)
    except Exception:  # noqa: BLE001
        return PreferenceContext(turn_overrides=turn_overrides)
    try:
        tenant_id = str(getattr(header, "tenant_id", "") or "").strip()
        user_id = int(getattr(header, "user_id", 0))
    except (TypeError, ValueError):
        return PreferenceContext(turn_overrides=turn_overrides)
    if not tenant_id or tenant_id == "None" or user_id <= 0:
        return PreferenceContext(turn_overrides=turn_overrides)
    try:
        from app.services.memory.memory_loader import MemoryLoader

        projection = await MemoryLoader.load_for_agent(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id="fin_agent",
        )
        preferences = dict(projection.as_dict())
    except Exception:  # noqa: BLE001
        preferences = {}
    return PreferenceContext(preferences=preferences, turn_overrides=turn_overrides)
