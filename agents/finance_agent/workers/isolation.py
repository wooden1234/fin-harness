"""Worker 冒泡隔离：同节点多实例并行时，中间态不写入父图共享 state。

LangGraph ``Send`` 扇出到同一 worker 时，各实例执行期互不共享内存；
但返回值会按字段 reducer 合并进父 ``FinAgentState``。对无 ``add`` 的
字段（如 ``financial_query_sql``）会后写覆盖先写。

因此父图节点只接受可安全并行合并的键；子 Agent 内部仍可读写完整中间态。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

# 父图可安全合并的字段（均带 list reducer 或由下游单点消费）
PARENT_SAFE_WORKER_KEYS = frozenset(
    {
        "messages",
        "task_results",
        "citations",
        "steps",
    }
)

WorkerFn = Callable[..., Awaitable[Any]]


def project_worker_updates_to_parent(updates: Mapping[str, Any]) -> dict[str, Any]:
    """过滤 worker 更新，仅保留可并行合并到父图的字段。"""
    return {
        key: value
        for key, value in updates.items()
        if key in PARENT_SAFE_WORKER_KEYS
    }


def isolate_worker_node(worker: Any) -> WorkerFn:
    """包装 worker，使其作为父图节点时不泄漏实例私有中间态。"""

    async def _isolated(
        state: Any,
        config: RunnableConfig = None,
    ) -> Any:
        if hasattr(worker, "ainvoke"):
            updates = await worker.ainvoke(state, config)
        else:
            updates = await worker(state, config)
        if not isinstance(updates, Mapping):
            return updates
        return project_worker_updates_to_parent(updates)

    return _isolated


__all__ = [
    "PARENT_SAFE_WORKER_KEYS",
    "isolate_worker_node",
    "project_worker_updates_to_parent",
]
