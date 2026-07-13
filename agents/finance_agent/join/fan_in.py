"""Fan-in 就绪判断：join 节点用 sub_tasks 与 task_results 判定是否可汇总。

用于 Map-Reduce 显式 join：所有子任务（含 faq/pdf → web 兜底）齐套后，
再由 join 条件边路由到 summarize。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def sub_task_satisfied(sub_task_id: str, results: Sequence[Mapping]) -> bool:
    """单个子任务是否已有可汇总结果（含 web 兜底完成）。"""
    related = [r for r in results if str(r.get("sub_task_id") or "") == sub_task_id]
    if not related:
        return False
    if any(r.get("fallback_to_web") for r in related):
        return any(
            r.get("type") == "web_search" and not r.get("fallback_to_web")
            for r in related
        )
    return True


def fan_in_ready(
    *,
    sub_tasks: Iterable[object],
    task_results: Sequence[Mapping],
) -> bool:
    """当前轮所有子任务是否都已满足汇总条件。

    ``sub_tasks`` 为空（澄清兜底）时视为就绪，由调用方直接汇总。
    """
    expected: set[str] = set()
    for task in sub_tasks or []:
        task_id = getattr(task, "id", None)
        if task_id is None and isinstance(task, Mapping):
            task_id = task.get("id")
        if task_id:
            expected.add(str(task_id))

    if not expected:
        return True

    return all(sub_task_satisfied(tid, task_results) for tid in expected)


__all__ = ["fan_in_ready", "sub_task_satisfied"]
