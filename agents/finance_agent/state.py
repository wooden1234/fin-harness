"""Finance Agent 领域 State：Planner 拆分 + Worker 输出汇总。

属于 finance_agent 大 Agent 的字段集中在此，
financial_query_agent 子图的状态在它自己的 state.py 里。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, NotRequired
from typing_extensions import TypedDict

from app.shared import Citation, SubTask, TaskResult


class PlannerState(TypedDict):
    """Planner 写入的多意图拆分结果"""
    planner_query: NotRequired[str]
    planner_raw_tasks: NotRequired[list[SubTask]]
    planner_validation_issues: NotRequired[list[str]]
    planner_needs_repair: NotRequired[bool]
    planner_repair_attempted: NotRequired[bool]
    planner_error_reason: NotRequired[str]
    sub_tasks: NotRequired[list[SubTask]]
    sub_question: NotRequired[str]
    sub_task_id: NotRequired[str]
    # Send 派发时随任务下发的证据降级链（如 ["faq", "web_search"]），
    # 供 worker 后条件边决定 uncovered 时的下一跳
    evidence_chain: NotRequired[list[str]]


class WorkerOutputState(TypedDict):
    """Worker 并行输出 + Summarize 汇总结果"""
    task_results: NotRequired[Annotated[list[TaskResult], add]]
    citations: NotRequired[Annotated[list[Citation], add]]
    summary: NotRequired[str]
