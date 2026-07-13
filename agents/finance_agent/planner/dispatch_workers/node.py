"""dispatch_workers 节点：派发 worker 与证据链降级路由（coverage gate）。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from agents.states import FinAgentState, SubTask
from agents.finance_agent.planner.common import CLARIFICATION_ANSWER, logger
from agents.finance_agent.planner.resolve_evidence import default_chain_for_type
from agents.finance_agent.planner.validate import ALLOWED_TASK_TYPES

TASK_TYPE_TO_WORKER = {
    "faq": "faq_agent",
    "pdf": "pdf_agent",
    "financial_query": "financial_query_agent",
    "web_search": "web_search_agent",
}


async def dispatch_workers_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """显式派发检查点：实际 Send 由 route_after_dispatch_workers 完成。"""
    del config
    sub_tasks = list(state.get("sub_tasks") or [])
    logger.info("dispatch_workers sub_tasks={}", len(sub_tasks))
    return {"steps": ["dispatch_workers"]}


def _task_chain(task: SubTask) -> list[str]:
    chain = list(getattr(task, "evidence_chain", None) or [])
    if chain:
        return chain
    return default_chain_for_type(str(task.type or ""))


def route_after_dispatch_workers(state: FinAgentState) -> list[Send]:
    """根据子任务的首选证据工具派发到 worker，并随任务下发降级链。"""
    sub_tasks: list[SubTask] = list(state.get("sub_tasks") or [])
    if not sub_tasks:
        return [
            Send(
                "join",
                {
                    "task_results": [
                        {
                            "sub_task_id": "",
                            "question": "",
                            "type": "planner_clarification",
                            "coverage": "clarify",
                            "context": CLARIFICATION_ANSWER,
                        }
                    ]
                },
            )
        ]

    sends: list[Send] = []
    for task in sub_tasks:
        worker = TASK_TYPE_TO_WORKER.get(task.type)
        if worker is None or task.type not in ALLOWED_TASK_TYPES:
            logger.warning("planner unknown task type={} question={}", task.type, task.question)
            sends.append(
                Send(
                    "join",
                    {
                        "task_results": [
                            {
                                "sub_task_id": task.id or "",
                                "question": task.question,
                                "type": str(task.type),
                                "coverage": "clarify",
                                "context": CLARIFICATION_ANSWER,
                            }
                        ]
                    },
                )
            )
            continue

        sends.append(
            Send(
                worker,
                {
                    "sub_question": task.question,
                    "sub_task_id": task.id or "",
                    "evidence_chain": _task_chain(task),
                },
            )
        )
    return sends


route_after_supervisor = route_after_dispatch_workers


def _result_unresolved(result: dict) -> bool:
    """worker 自评证据不足（coverage=uncovered 或旧 fallback_to_web）。"""
    if result.get("fallback_to_web"):
        return True
    return str(result.get("coverage") or "") == "uncovered"


def _next_tool_in_chain(current_tool: str, chain: list[str]) -> str | None:
    """返回链上 current_tool 的下一跳；不在链上时按旧行为兜底 web。"""
    if chain and current_tool in chain:
        idx = chain.index(current_tool)
        return chain[idx + 1] if idx + 1 < len(chain) else None
    # 旧数据兜底：faq/pdf 检索失败默认联网
    if current_tool in {"faq", "pdf"}:
        return "web_search"
    return None


def route_after_retrieval_worker(state: FinAgentState) -> str | Send:
    """coverage gate：证据不足时沿降级链下一跳，否则收敛到 join。

    降级必须用 ``Send`` 显式带上 ``sub_task_id`` / ``sub_question`` /
    ``evidence_chain``：上游 ``Send`` 的字段只在 worker 执行期可见，
    普通边不会自动带到下一跳。
    """
    sub_task_id = str(state.get("sub_task_id") or "")
    sub_question = str(state.get("sub_question") or "")
    chain = list(state.get("evidence_chain") or [])
    task_results = list(state.get("task_results") or [])

    for result in reversed(task_results):
        result_id = str(result.get("sub_task_id") or "")
        if sub_task_id and result_id != sub_task_id:
            continue
        if _result_unresolved(result):
            current_tool = str(result.get("type") or "")
            next_tool = _next_tool_in_chain(current_tool, chain)
            next_worker = TASK_TYPE_TO_WORKER.get(next_tool or "")
            if next_worker:
                logger.info(
                    "coverage gate: task={} tool={} uncovered → {}",
                    result_id or sub_task_id,
                    current_tool,
                    next_tool,
                )
                return Send(
                    next_worker,
                    {
                        "sub_task_id": result_id or sub_task_id,
                        "sub_question": sub_question or str(result.get("question") or ""),
                        "evidence_chain": chain,
                    },
                )
        return "join"

    return "join"


__all__ = [
    "TASK_TYPE_TO_WORKER",
    "dispatch_workers_node",
    "route_after_dispatch_workers",
    "route_after_retrieval_worker",
    "route_after_supervisor",
]
