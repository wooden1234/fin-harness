"""记忆校验用的静态 Agent 规格，不依赖 Orchestrator。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryAgentSpec:
    agent_id: str
    memory_keys: tuple[str, ...]
    semantic_memory_types: tuple[str, ...] = ()


_COMMON = (
    "response_language",
    "response_detail_level",
    "preferred_output_format",
)

_FINANCE = _COMMON + (
    "default_currency",
    "default_market",
    "default_compare_period",
    "citation_preference",
)

PRODUCT_MEMORY_SPECS: tuple[MemoryAgentSpec, ...] = (
    MemoryAgentSpec("fin_agent", _FINANCE, ("episodic",)),
    MemoryAgentSpec("general_agent", _COMMON, ("episodic",)),
    MemoryAgentSpec("finance_agent", _FINANCE, ("episodic",)),
)

_BY_ID = {item.agent_id: item for item in PRODUCT_MEMORY_SPECS}


def list_memory_agent_specs() -> list[MemoryAgentSpec]:
    return list(PRODUCT_MEMORY_SPECS)


def get_agent_spec(agent_id: str) -> MemoryAgentSpec:
    try:
        return _BY_ID[agent_id]
    except KeyError as exc:
        raise ValueError(f"agent_not_registered:{agent_id}") from exc
