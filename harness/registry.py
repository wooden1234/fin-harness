"""graph / skill / tool 的统一注册入口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.errors import RegistryLookupError

RegistryCallable = Callable[..., Any]

_GRAPHS: dict[str, RegistryCallable] = {}
_SKILLS: dict[str, RegistryCallable] = {}
_TOOLS: dict[str, RegistryCallable] = {}


def register_graph(name: str, target: RegistryCallable) -> None:
    _GRAPHS[name] = target


def register_skill(name: str, target: RegistryCallable) -> None:
    _SKILLS[name] = target


def register_tool(name: str, target: RegistryCallable) -> None:
    _TOOLS[name] = target


def get_graph(name: str) -> RegistryCallable:
    try:
        return _GRAPHS[name]
    except KeyError as exc:
        raise RegistryLookupError(f"graph not registered: {name}") from exc


def list_registered() -> dict[str, list[str]]:
    return {
        "graphs": sorted(_GRAPHS),
        "skills": sorted(_SKILLS),
        "tools": sorted(_TOOLS),
    }
