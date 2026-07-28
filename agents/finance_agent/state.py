"""Finance Agent 领域 State：Planner 拆分 + Worker 输出汇总。

属于 finance_agent 大 Agent 的字段集中在此，
financial_query_agent 子图的状态在它自己的 state.py 里。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, NotRequired
from typing_extensions import TypedDict

from app.shared import Citation, SubTask, TaskResult


class PlannerState(TypedDict):
    """Planner 写入的多意图拆分结果"""
    # Root 传入的领域规划权限边界；运行时使用统一 DomainPlanningScope 校验。
    domain_planning_scope: NotRequired[Any]
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
    scope_blocked_task_ids: NotRequired[list[str]]


class WorkerOutputState(TypedDict):
    """Worker 并行输出 + Summarize 汇总结果"""
    # Root Orchestrator 传入的完整依赖，不与用户问题拼接。
    dependency_results: NotRequired[list[Any]]
    task_input: NotRequired[dict[str, Any]]
    task_results: NotRequired[Annotated[list[TaskResult], add]]
    # 兼容迁移期间的统一结果；旧 task_results 暂时继续保留。
    # 使用 Any 避免领域 State 反向导入 Root Orchestrator 造成初始化环；
    # 运行时实际写入的是 agents.orchestrator.contracts.AgentResult。
    agent_results: NotRequired[Annotated[list[Any], add]]
    citations: NotRequired[Annotated[list[Citation], add]]
    # 当前一轮金融任务的候选答案；final_answer 采用后清空。
    # 切勿与 conversation_summary（多轮会话压缩记忆）混用。
    summary: NotRequired[str]
    rag_trace: NotRequired[dict[str, Any]]
