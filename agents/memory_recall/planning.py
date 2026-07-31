"""任务级记忆需求规划与只读投影加载。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.state import OrchestratorState
from agents.runtime_context import AgentRuntimeContext
from app.core.logger import get_logger
from app.core.config import settings
from app.services.memory.memory_command import extract_turn_preferences
from app.services.memory.memory_audit import MemoryAuditContext
from app.services.memory.memory_loader import MemoryLoader

logger = get_logger(service="memory_planning")


def _latest_query(state: OrchestratorState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def memory_plan_node(
    state: OrchestratorState,
) -> dict[str, Any]:
    """根据已验证 TaskPlan 为每个任务生成静态记忆需求。"""
    plan = state.get("task_plan")
    requirements: dict[str, dict[str, Any]] = {}
    if plan is not None:
        for task in plan.tasks:
            spec = get_agent_spec(task.agent_id)
            semantic_memory_type = ""
            if task.semantic_history:
                if task.semantic_memory_type not in spec.semantic_memory_types:
                    raise PermissionError(
                        "agent_semantic_memory_type_denied:"
                        f"{task.agent_id}:{task.semantic_memory_type}"
                    )
                semantic_memory_type = task.semantic_memory_type
            requirements[task.task_id] = {
                "agent_id": task.agent_id,
                "memory_keys": list(spec.memory_keys),
                "semantic_memory_type": semantic_memory_type,
                "semantic_query": (
                    task.objective if semantic_memory_type else ""
                ),
                "semantic_top_k": settings.MEMORY_RECALL_TOP_K,
            }
    return {
        "memory_requirements": requirements,
        "turn_preferences": extract_turn_preferences(_latest_query(state)),
        "steps": [f"memory:plan:{len(requirements)}"],
    }


def _requirement_signature(
    requirement: dict[str, Any],
) -> tuple[str, tuple[str, ...], str, str, int]:
    return (
        str(requirement.get("agent_id") or ""),
        tuple(str(key) for key in requirement.get("memory_keys") or []),
        str(requirement.get("semantic_memory_type") or ""),
        str(requirement.get("semantic_query") or ""),
        int(requirement.get("semantic_top_k") or 0),
    )


async def load_task_memories_node(
    state: OrchestratorState,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """集中加载任务记忆；相同白名单及 replan 任务复用既有投影。"""
    requirements = dict(state.get("memory_requirements") or {})
    previous_contexts = dict(state.get("task_memory_context") or {})
    contexts = {
        task_id: deepcopy(previous_contexts[task_id])
        for task_id in requirements
        if task_id in previous_contexts
    }
    signature_contexts: dict[
        tuple[str, tuple[str, ...], str, str, int],
        dict[str, Any],
    ] = {}
    for task_id, context in contexts.items():
        requirement = requirements.get(task_id)
        if requirement is not None:
            signature_contexts[_requirement_signature(requirement)] = deepcopy(context)

    pending: dict[
        tuple[str, tuple[str, ...], str, str, int],
        list[str],
    ] = {}
    for task_id, requirement in requirements.items():
        if task_id in contexts:
            continue
        signature = _requirement_signature(requirement)
        cached = signature_contexts.get(signature)
        if cached is not None:
            contexts[task_id] = deepcopy(cached)
            continue
        pending.setdefault(signature, []).append(task_id)

    runtime_context = runtime.context if runtime is not None else None
    turn_preferences = dict(state.get("turn_preferences") or {})
    bypass_cache = bool(state.get("memory_cache_bypass"))

    async def load_signature(
        signature: tuple[str, tuple[str, ...], str, str, int],
        task_id: str,
    ) -> tuple[
        tuple[str, tuple[str, ...], str, str, int],
        dict[str, Any],
    ]:
        (
            agent_id,
            memory_keys,
            semantic_memory_type,
            semantic_query,
            semantic_top_k,
        ) = signature
        if runtime_context is None or (
            not memory_keys and not semantic_memory_type
        ):
            return signature, {}
        try:
            audit_context = MemoryAuditContext(
                tenant_id=str(runtime_context.tenant_id),
                user_id=int(runtime_context.user_id),
                agent_id=agent_id,
                task_id=task_id,
                trace_id=getattr(runtime_context, "run_id", None),
            )
            effective: dict[str, Any] = {}
            if memory_keys:
                projection = await MemoryLoader.load_for_agent(
                    tenant_id=runtime_context.tenant_id,
                    user_id=int(runtime_context.user_id),
                    agent_id=agent_id,
                    memory_keys=memory_keys,
                    bypass_cache=bypass_cache,
                    audit_context=audit_context,
                )
                effective.update(projection.as_dict())
            if semantic_memory_type:
                semantic_projection = (
                    await MemoryLoader.load_semantic_for_agent(
                        tenant_id=runtime_context.tenant_id,
                        user_id=int(runtime_context.user_id),
                        agent_id=agent_id,
                        memory_type=semantic_memory_type,
                        query=semantic_query,
                        top_k=semantic_top_k,
                        audit_context=audit_context,
                    )
                )
                effective["semantic_history"] = (
                    semantic_projection.as_list()
                )
            for memory_key in memory_keys:
                if memory_key in turn_preferences:
                    effective[memory_key] = deepcopy(
                        turn_preferences[memory_key]
                    )
            return signature, effective
        except Exception as exc:
            logger.warning(
                "task memory load failed; continue without memory: agent={} error={}",
                agent_id,
                type(exc).__name__,
            )
            return signature, {}

    loaded = await asyncio.gather(
        *(
            load_signature(signature, pending[signature][0])
            for signature in pending
        )
    )
    for signature, context in loaded:
        signature_contexts[signature] = deepcopy(context)
        for task_id in pending[signature]:
            contexts[task_id] = deepcopy(context)

    return {
        "task_memory_context": contexts,
        "steps": [
            f"memory:load:tasks={len(requirements)}:queries={len(pending)}"
        ],
    }


__all__ = ["load_task_memories_node", "memory_plan_node"]
