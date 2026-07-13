"""resolve_evidence 节点：意图 → 证据工具降级链。

映射写死在代码而非交给 LLM：知识库/文档/SQL/联网只是证据渠道，
覆盖不到时按链条降级，而不是让模型硬答。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState, SubTask
from app.core.logger import get_logger

logger = get_logger(service="resolve_evidence")

# intent → 有序证据链；链首为首选工具，后续为 uncovered 时的降级跳
INTENT_TO_EVIDENCE_CHAIN: dict[str, list[str]] = {
    "concept_explain": ["faq", "web_search"],
    "product_policy": ["faq", "web_search"],
    "document_qa": ["pdf", "web_search"],
    # 数字类禁止降级到 faq 编数；SQL 查无时由联网补充公开口径并标注来源
    "structured_metric": ["financial_query", "web_search"],
    "market_event": ["web_search"],
}

# 兜底：仅有 type（旧数据/直连派发）时的默认链
_TYPE_TO_DEFAULT_CHAIN: dict[str, list[str]] = {
    "faq": ["faq", "web_search"],
    "pdf": ["pdf", "web_search"],
    "financial_query": ["financial_query", "web_search"],
    "web_search": ["web_search"],
}


def default_chain_for_type(task_type: str) -> list[str]:
    return list(_TYPE_TO_DEFAULT_CHAIN.get(task_type, [task_type] if task_type else []))


def resolve_task_evidence(task: SubTask) -> SubTask:
    """按意图填充首选证据工具与降级链；无意图时按 type 兜底。"""
    intent = str(getattr(task, "intent", "") or "")
    chain = list(INTENT_TO_EVIDENCE_CHAIN.get(intent) or [])
    if not chain:
        chain = default_chain_for_type(str(task.type or ""))
    if not chain:
        chain = ["faq", "web_search"]
    return SubTask(
        id=task.id,
        question=task.question,
        intent=intent,
        type=chain[0],
        evidence_chain=chain,
    )


async def resolve_evidence_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """确定性节点：为每个子任务填充证据链，不调用 LLM。"""
    del config
    sub_tasks = [resolve_task_evidence(task) for task in (state.get("sub_tasks") or [])]
    logger.info(
        "resolve_evidence tasks={} chains={}",
        len(sub_tasks),
        [(t.intent or t.type, t.evidence_chain) for t in sub_tasks],
    )
    return {"sub_tasks": sub_tasks, "steps": ["resolve_evidence"]}


__all__ = [
    "INTENT_TO_EVIDENCE_CHAIN",
    "default_chain_for_type",
    "resolve_evidence_node",
    "resolve_task_evidence",
]
