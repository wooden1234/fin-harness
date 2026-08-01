"""resolve_evidence 节点：意图 → 证据工具降级链。

映射写死在代码而非交给 LLM：知识库/文档/SQL/联网只是证据渠道，
覆盖不到时按链条降级，而不是让模型硬答。
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState, SubTask
from agents.finance_agent.planner.scope import domain_scope_from_state
from app.core.logger import get_logger

logger = get_logger(service="resolve_evidence")

# intent → 有序证据链；链首为首选工具，后续为 uncovered 时的降级跳
INTENT_TO_EVIDENCE_CHAIN: dict[str, list[str]] = {
    "concept_explain": ["faq", "web_search"],
    "product_policy": ["faq", "web_search"],
    "document_qa": ["pdf", "web_search"],
    # 数字类禁止降级到 faq 编数；SQL 查无时先回年报，再由联网补充公开口径
    "structured_metric": ["financial_query", "pdf", "web_search"],
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


def resolve_task_evidence(
    task: SubTask,
    *,
    allowed_capabilities: set[str] | None = None,
    allowed_intents: set[str] | None = None,
    fallback_policy: str = "within_scope",
    allow_web_fallback: bool = True,
) -> SubTask:
    """按意图填充首选证据工具与降级链；无意图时按 type 兜底。"""
    intent = str(getattr(task, "intent", "") or "")
    chain = list(INTENT_TO_EVIDENCE_CHAIN.get(intent) or [])
    if not chain:
        chain = default_chain_for_type(str(task.type or ""))
    if not chain:
        chain = ["faq", "web_search"]
    if not allow_web_fallback and chain[:1] != ["web_search"]:
        chain = [item for item in chain if item != "web_search"]
    if allowed_capabilities is not None:
        if allowed_intents is not None and intent not in allowed_intents:
            chain = []
        elif fallback_policy == "deny":
            chain = [
                chain[0]
            ] if chain and chain[0] in allowed_capabilities else []
        else:
            chain = [
                capability
                for capability in chain
                if capability in allowed_capabilities
            ]
    return SubTask(
        id=task.id,
        question=task.question,
        intent=intent,
        reason=str(getattr(task, "reason", "") or "").strip(),
        type=chain[0] if chain else task.type,
        evidence_chain=chain,
    )


async def resolve_evidence_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """确定性节点：为每个子任务填充证据链，不调用 LLM。"""
    del config
    scope_present = (
        state.get("domain_planning_scope") is not None
        or (
            isinstance(state.get("task_input"), dict)
            and state["task_input"].get("domain_planning_scope") is not None
        )
    )
    try:
        domain_scope = domain_scope_from_state(state)
    except (TypeError, ValueError):
        domain_scope = None
        scope_present = True
    allowed_capabilities = (
        set(domain_scope.allowed_capabilities)
        if domain_scope is not None
        else (set() if scope_present else None)
    )
    allowed_intents = (
        set(domain_scope.allowed_intents)
        if domain_scope is not None
        else (set() if scope_present else None)
    )
    fallback_policy = (
        domain_scope.fallback_policy
        if domain_scope is not None
        else "within_scope"
    )
    allow_web_fallback = (
        domain_scope is None or domain_scope.freshness_required
    )
    sub_tasks = [
        resolve_task_evidence(
            task,
            allowed_capabilities=allowed_capabilities,
            allowed_intents=allowed_intents,
            fallback_policy=fallback_policy,
            allow_web_fallback=allow_web_fallback,
        )
        for task in (state.get("sub_tasks") or [])
    ]
    blocked_ids = [
        task.id
        for task in sub_tasks
        if scope_present and not task.evidence_chain
    ]
    logger.info(
        "resolve_evidence tasks={} chains={}",
        len(sub_tasks),
        [(t.intent or t.type, t.evidence_chain) for t in sub_tasks],
    )
    return {
        "sub_tasks": sub_tasks,
        "scope_blocked_task_ids": blocked_ids,
        "steps": ["resolve_evidence"],
    }


__all__ = [
    "INTENT_TO_EVIDENCE_CHAIN",
    "default_chain_for_type",
    "resolve_evidence_node",
    "resolve_task_evidence",
]
