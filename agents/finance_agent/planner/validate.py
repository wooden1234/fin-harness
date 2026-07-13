"""Planner 输出确定性校验与规范化。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.shared import SubTask

ALLOWED_TASK_TYPES = frozenset({"faq", "pdf", "financial_query", "web_search"})
MAX_SUBTASKS = 4

_WS_RE = re.compile(r"\s+")


@dataclass
class ValidationResult:
    """规范化后的子任务 + 问题列表。"""

    tasks: list[SubTask] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    needs_repair: bool = False


def _normalize_question(question: str) -> str:
    return _WS_RE.sub("", (question or "").strip().lower())


def _is_near_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter not in longer:
        return False
    return len(shorter) / len(longer) >= 0.8


def validate_and_normalize_tasks(tasks: list[SubTask] | None) -> ValidationResult:
    """丢弃空问句 / 禁止类型，合并同 type 近义，截断超拆。

    ``needs_repair``：出现无法仅靠确定性规则放心留下的脏数据
    （空 question、forbidden/unknown type）。合并与截断视为软修复，不强制 repair。
    """
    issues: list[str] = []
    needs_repair = False
    cleaned: list[SubTask] = []

    for task in tasks or []:
        question = (task.question or "").strip()
        if not question:
            issues.append("empty_question")
            needs_repair = True
            continue

        task_type = str(task.type or "").strip()
        if task_type == "general":
            issues.append("forbidden_type:general")
            needs_repair = True
            continue
        if task_type not in ALLOWED_TASK_TYPES:
            issues.append(f"unknown_type:{task_type or '<empty>'}")
            needs_repair = True
            continue

        cleaned.append(SubTask(id=task.id or "", question=question, type=task_type))

    merged: list[SubTask] = []
    for task in cleaned:
        norm = _normalize_question(task.question)
        duplicate_idx = next(
            (
                i
                for i, kept in enumerate(merged)
                if kept.type == task.type
                and _is_near_duplicate(norm, _normalize_question(kept.question))
            ),
            None,
        )
        if duplicate_idx is None:
            merged.append(task)
            continue
        issues.append(f"merged_duplicate:{task.type}")
        kept = merged[duplicate_idx]
        if len(task.question) > len(kept.question):
            merged[duplicate_idx] = SubTask(
                id=kept.id or task.id,
                question=task.question,
                type=task.type,
            )

    if len(merged) > MAX_SUBTASKS:
        issues.append(f"exceeds_max:{len(merged)}>{MAX_SUBTASKS}")
        merged = merged[:MAX_SUBTASKS]

    return ValidationResult(tasks=merged, issues=issues, needs_repair=needs_repair)
