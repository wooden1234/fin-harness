"""Skill 注册表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SkillCallable = Callable[..., Any]

_SKILLS: dict[str, SkillCallable] = {}


def register_skill(name: str, target: SkillCallable) -> None:
    _SKILLS[name] = target


def get_skill(name: str) -> SkillCallable:
    return _SKILLS[name]


def list_skills() -> list[str]:
    return sorted(_SKILLS)
