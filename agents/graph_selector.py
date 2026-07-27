"""V1/V2 Graph 选择器，保持旧 Graph 实现不变。"""

from __future__ import annotations

import hashlib
from importlib import import_module
from functools import lru_cache

from app.core.config import settings


def _allowlist() -> frozenset[str]:
    return frozenset(
        item.strip()
        for item in settings.AGENT_GRAPH_V2_TENANT_ALLOWLIST.split(",")
        if item.strip()
    )


def _stable_bucket(identity: str) -> int:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def select_graph_version(
    *,
    conversation_id: str | int,
    user_id: str | int | None = None,
    tenant_id: str | int | None = None,
) -> str:
    """根据配置和稳定身份选择图版本。"""
    mode = str(settings.AGENT_GRAPH_MODE or "v2").strip().lower()
    if mode not in {"v1", "v2", "rollout"}:
        raise ValueError(f"invalid_agent_graph_mode:{mode}")
    if mode == "v1":
        return "v1"
    if mode == "v2":
        return "v2"

    tenant_text = str(tenant_id) if tenant_id is not None else ""
    if tenant_text and tenant_text in _allowlist():
        return "v2"

    # 未建立版本字段前，存量整数会话固定走 V1，避免切换图后读取不兼容的 Checkpoint。
    if settings.AGENT_GRAPH_V2_NEW_CONVERSATIONS_ONLY and isinstance(conversation_id, int):
        return "v1"

    identity = f"{tenant_id or 'anonymous'}:{user_id or 'anonymous'}:{conversation_id}"
    percent = max(0, min(100, int(settings.AGENT_GRAPH_V2_ROLLOUT_PERCENT)))
    return "v2" if _stable_bucket(identity) < percent else "v1"


@lru_cache(maxsize=2)
def _get_graph(version: str, with_checkpointer: bool):
    if version == "v1":
        return import_module("agent-v1.graph").get_graph(
            with_checkpointer=with_checkpointer
        )
    if version == "v2":
        from agents.orchestrator.graph import get_orchestrator_graph

        return get_orchestrator_graph(with_checkpointer=with_checkpointer)
    raise ValueError(f"unknown_agent_graph_version:{version}")


def get_selected_graph(version: str, *, with_checkpointer: bool = True):
    """获取指定版本图；版本由调用方先通过 select_graph_version 决定。"""
    return _get_graph(version, with_checkpointer)


def reset_graph_selector_cache() -> None:
    _get_graph.cache_clear()


__all__ = [
    "get_selected_graph",
    "reset_graph_selector_cache",
    "select_graph_version",
]
