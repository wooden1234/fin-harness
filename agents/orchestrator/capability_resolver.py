"""根据请求画像确定性推导 Root 任务所需能力。"""

from __future__ import annotations

from collections.abc import Iterable

from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.contracts import RequestProfile


def _registered_finance_capabilities() -> set[str]:
    return set(get_agent_spec("finance_agent").capabilities)


def resolve_finance_capabilities(
    profile: RequestProfile,
    *,
    allowed_capabilities: Iterable[str] = (),
) -> list[str]:
    """把确定性 Resolver 的 Finance 授权与注册能力、领域范围求交。"""
    registered = _registered_finance_capabilities()
    allowed = set(allowed_capabilities) or registered
    return list(dict.fromkeys(
        capability
        for capability in profile.execution.allowed_capabilities
        if capability in registered and capability in allowed
    ))


__all__ = ["resolve_finance_capabilities"]
