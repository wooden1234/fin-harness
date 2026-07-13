"""Planner 输出确定性校验与规范化（意图维度）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.shared import SubTask

ALLOWED_INTENTS = frozenset(
    {
        "concept_explain",
        "product_policy",
        "document_qa",
        "structured_metric",
        "market_event",
    }
)
ALLOWED_TASK_TYPES = frozenset({"faq", "pdf", "financial_query", "web_search"})
MAX_SUBTASKS = 4

# 兼容旧 planner 输出（type 维度）：按数据源反推最接近的意图
_LEGACY_TYPE_TO_INTENT = {
    "faq": "concept_explain",
    "pdf": "document_qa",
    "financial_query": "structured_metric",
    "web_search": "market_event",
}

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


def _resolve_intent(task: SubTask, issues: list[str]) -> str | None:
    """解析任务意图；返回 None 表示该任务需要 repair。"""
    intent = str(getattr(task, "intent", "") or "").strip()
    if intent in ALLOWED_INTENTS:
        return intent

    task_type = str(task.type or "").strip()
    if intent:
        issues.append(f"unknown_intent:{intent}")
        return None
    # LLM 未输出 intent（如旧输出格式）：从 type 反推
    if task_type in _LEGACY_TYPE_TO_INTENT:
        return _LEGACY_TYPE_TO_INTENT[task_type]
    if task_type == "general":
        issues.append("forbidden_type:general")
        return None
    issues.append(f"unknown_type:{task_type or '<empty>'}")
    return None


def validate_and_normalize_tasks(tasks: list[SubTask] | None) -> ValidationResult:
    """丢弃空问句 / 非法意图，合并同意图近义，截断超拆。

    ``needs_repair``：出现无法仅靠确定性规则放心留下的脏数据
    （空 question、非法 intent/type）。合并与截断视为软修复，不强制 repair。
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

        intent = _resolve_intent(task, issues)
        if intent is None:
            needs_repair = True
            continue

        cleaned.append(SubTask(id=task.id or "", question=question, intent=intent))

    merged: list[SubTask] = []
    for task in cleaned:
        norm = _normalize_question(task.question)
        duplicate_idx = next(
            (
                i
                for i, kept in enumerate(merged)
                if kept.intent == task.intent
                and _is_near_duplicate(norm, _normalize_question(kept.question))
            ),
            None,
        )
        if duplicate_idx is None:
            merged.append(task)
            continue
        issues.append(f"merged_duplicate:{task.intent}")
        kept = merged[duplicate_idx]
        if len(task.question) > len(kept.question):
            merged[duplicate_idx] = SubTask(
                id=kept.id or task.id,
                question=task.question,
                intent=task.intent,
            )

    if len(merged) > MAX_SUBTASKS:
        issues.append(f"exceeds_max:{len(merged)}>{MAX_SUBTASKS}")
        merged = merged[:MAX_SUBTASKS]

    return ValidationResult(tasks=merged, issues=issues, needs_repair=needs_repair)
