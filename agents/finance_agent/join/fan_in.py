"""Fan-in 就绪判断：join 节点用 sub_tasks 与 task_results 判定是否可汇总。

用于 Map-Reduce 显式 join：某个子任务的结果 coverage=uncovered 且证据链
未走完时，视为未齐（等待降级跳的结果）；链走完或已有可用结果则就绪。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def _result_unresolved(result: Mapping) -> bool:
    if result.get("fallback_to_web"):
        return True
    return str(result.get("coverage") or "") == "uncovered"


def _next_tool_in_chain(current_tool: str, chain: Sequence[str] | None) -> str | None:
    if chain and current_tool in chain:
        idx = list(chain).index(current_tool)
        return chain[idx + 1] if idx + 1 < len(chain) else None
    # 旧数据兜底：faq/pdf 检索失败默认还有 web 一跳
    if current_tool in {"faq", "pdf"}:
        return "web_search"
    return None


def sub_task_satisfied(
    sub_task_id: str,
    results: Sequence[Mapping],
    chain: Sequence[str] | None = None,
) -> bool:
    """单个子任务是否已有可汇总结果（证据链降级完成也算）。"""
    related = [r for r in results if str(r.get("sub_task_id") or "") == sub_task_id]
    if not related:
        return False
    for result in related:
        if not _result_unresolved(result):
            return True
        # uncovered 但链已走完：作为「无可靠依据」的终态交给 summarize
        if _next_tool_in_chain(str(result.get("type") or ""), chain) is None:
            return True
    return False


def fan_in_ready(
    *,
    sub_tasks: Iterable[object],
    task_results: Sequence[Mapping],
) -> bool:
    """当前轮所有子任务是否都已满足汇总条件。

    ``sub_tasks`` 为空（澄清兜底）时视为就绪，由调用方直接汇总。
    """
    expected: dict[str, Sequence[str] | None] = {}
    for task in sub_tasks or []:
        task_id = getattr(task, "id", None)
        chain = getattr(task, "evidence_chain", None)
        if task_id is None and isinstance(task, Mapping):
            task_id = task.get("id")
            chain = task.get("evidence_chain")
        if task_id:
            expected[str(task_id)] = list(chain) if chain else None

    if not expected:
        return True

    return all(
        sub_task_satisfied(tid, task_results, chain=chain)
        for tid, chain in expected.items()
    )


__all__ = ["fan_in_ready", "sub_task_satisfied"]
