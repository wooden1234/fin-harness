"""dispatch_workers 节点：派发 worker 与检索兜底路由。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from agents.states import FinAgentState, SubTask
from agents.finance_agent.planner.common import CLARIFICATION_ANSWER, logger
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


def route_after_dispatch_workers(state: FinAgentState) -> list[Send]:
    """根据 planner 产出的子任务类型直接派发到 worker。"""
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
                },
            )
        )
    return sends


route_after_supervisor = route_after_dispatch_workers


def route_after_retrieval_worker(state: FinAgentState) -> str | Send:
    """根据检索结果决定是否进入联网搜索兜底。

    兜底必须用 ``Send`` 显式带上 ``sub_task_id`` / ``sub_question``：
    上游 ``Send`` 的字段只在 worker 执行期可见，普通边不会自动带到下一跳。
    """
    sub_task_id = str(state.get("sub_task_id") or "")
    sub_question = str(state.get("sub_question") or "")
    task_results = list(state.get("task_results") or [])

    for result in reversed(task_results):
        result_id = str(result.get("sub_task_id") or "")
        if sub_task_id and result_id != sub_task_id:
            continue
        if result.get("fallback_to_web"):
            return Send(
                "web_search_agent",
                {
                    "sub_task_id": result_id or sub_task_id,
                    "sub_question": sub_question or str(result.get("question") or ""),
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
