"""Join 节点：Map-Reduce 中的显式收齐点。

所有 worker（含 faq/pdf 兜底 web）完成后才路由到 summarize；
未齐的分支在 join 处结束，避免 summarize 被短路径空跑触发。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from agents.finance_agent.join.fan_in import fan_in_ready
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="join")


async def join_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """Fan-in 检查点：不修改业务 state，仅记录收齐进度。"""
    sub_tasks = list(state.get("sub_tasks") or [])
    task_results = list(state.get("task_results") or [])
    ready = fan_in_ready(sub_tasks=sub_tasks, task_results=task_results)

    if ready:
        logger.info(
            "join ready: sub_tasks={} results={}",
            len(sub_tasks),
            len(task_results),
        )
    else:
        logger.info(
            "join waiting: sub_tasks={} results={}",
            len(sub_tasks),
            len(task_results),
        )

    return {"steps": ["join"]}


def route_after_join(state: FinAgentState) -> str:
    """齐套 → summarize；未齐 → 结束本分支，等待其它并行路径。"""
    if fan_in_ready(
        sub_tasks=state.get("sub_tasks") or [],
        task_results=state.get("task_results") or [],
    ):
        return "summarize"
    return END


__all__ = ["join_node", "route_after_join"]
