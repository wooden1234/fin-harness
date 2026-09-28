"""每步加载长期偏好投影与本轮临时覆盖；失败空结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PreferenceContext:
    preferences: dict[str, Any] = field(default_factory=dict)
    turn_overrides: dict[str, str] = field(default_factory=dict)
    session_overrides: dict[str, str] = field(default_factory=dict)

    def effective(self) -> dict[str, Any]:
        """Return the preferences that every user-visible generator must obey."""
        merged = dict(self.preferences)
        merged.update(self.session_overrides)
        merged.update(self.turn_overrides)
        return merged


def _user_message_text(event: Any, *, turn: int | None = None) -> str | None:
    if turn is not None and getattr(event, "turn", None) != turn:
        return None
    if getattr(event, "event_type", None) != "user/message":
        return None
    data = getattr(event, "data", None) or {}
    if not isinstance(data, Mapping):
        return None
    if str(data.get("source") or "user") != "user":
        return None
    return str(data.get("content") or "")


def _collected_preferences(events: Sequence[Any], extractor: Any, *, turn: int | None = None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for event in events:
        text = _user_message_text(event, turn=turn)
        if text:
            merged.update(extractor(text))
    return merged


def _turn_overrides(events: Sequence[Any], turn: int) -> dict[str, str]:
    try:
        from app.services.memory.memory_command import extract_turn_preferences
    except Exception:  # noqa: BLE001
        return {}
    return _collected_preferences(events, extract_turn_preferences, turn=turn)


def _session_overrides(events: Sequence[Any]) -> dict[str, str]:
    try:
        from app.services.memory.memory_command import extract_session_preferences
    except Exception:  # noqa: BLE001
        return {}
    return _collected_preferences(events, extract_session_preferences)


async def load_preference_context(
    *,
    store: Any,
    session_id: str,
    events: Sequence[Any],
    turn: int,
) -> PreferenceContext:
    """从 session header 加载 fin_agent 白名单偏好；任何失败都返回空偏好。"""
    turn_overrides = _turn_overrides(events, turn)
    session_overrides = _session_overrides(events)
    try:
        header = await store.get(session_id)
    except Exception:  # noqa: BLE001
        return PreferenceContext(
            turn_overrides=turn_overrides,
            session_overrides=session_overrides,
        )
    try:
        tenant_id = str(getattr(header, "tenant_id", "") or "").strip()
        user_id = int(getattr(header, "user_id", 0))
    except (TypeError, ValueError):
        return PreferenceContext(
            turn_overrides=turn_overrides,
            session_overrides=session_overrides,
        )
    if not tenant_id or tenant_id == "None" or user_id <= 0:
        return PreferenceContext(
            turn_overrides=turn_overrides,
            session_overrides=session_overrides,
        )
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
    return PreferenceContext(
        preferences=preferences,
        turn_overrides=turn_overrides,
        session_overrides=session_overrides,
    )
