"""Agent 图入口的长期偏好召回节点。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState
from app.core.logger import get_logger
from app.services.memory.memory_command import (
    extract_turn_preferences,
    parse_memory_rule_action,
)
from app.services.memory.memory_episodic_extraction import (
    decide_post_turn_trigger,
    detect_important_state_change,
    estimate_episodic_tokens,
    is_substantive_progress,
)
from app.services.memory.memory_recall import recall_preferences
from app.services.persistence import OutboxService

logger = get_logger(service="memory_recall")


def _latest_query(state: FinAgentState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def memory_recall_node(
    state: FinAgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict:
    context = runtime.context
    query = _latest_query(state)
    turn_preferences = extract_turn_preferences(query)
    if context is None:
        return {
            "memory_context": {},
            "turn_preferences": turn_preferences,
        }
    try:
        preferences = await recall_preferences(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
            query=query,
        )
    except Exception:
        logger.exception("memory recall failed; continue without long-term memory")
        preferences = {}
    return {
        "memory_context": preferences,
        "turn_preferences": turn_preferences,
    }


def _latest_assistant_message(state: FinAgentState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, AIMessage):
            return str(message.content or "")
    return ""


def _task_count(state: FinAgentState) -> int:
    task_plan = state.get("task_plan")
    if task_plan is not None:
        return len(list(getattr(task_plan, "tasks", []) or []))
    return len(list(state.get("agent_results") or []))


async def post_turn_memory_node(
    state: FinAgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, object]:
    """回答完成后登记后台记忆任务；不阻塞本轮回答。"""
    context = runtime.context
    if context is None or not context.run_id:
        return {"post_turn_memory_status": "skipped_no_runtime"}
    if state.get("guardrails_pass") is False:
        return {"post_turn_memory_status": "skipped_guardrail"}
    if state.get("route") == "memory_management":
        return {"post_turn_memory_status": "skipped_memory_management"}

    query = _latest_query(state)
    final_response = _latest_assistant_message(state)
    if not query or not final_response:
        return {"post_turn_memory_status": "skipped_empty_turn"}

    action = parse_memory_rule_action(query)
    enqueued: list[str] = []
    try:
        if action.kind == "implicit":
            await OutboxService.enqueue_memory_extraction(
                run_id=context.run_id,
                user_id=int(context.user_id),
                tenant_id=str(context.tenant_id),
                conversation_id=(
                    int(context.conversation_id)
                    if context.conversation_id is not None
                    else None
                ),
                source_text=query,
                agent_id="orchestrator",
                task_id="memory-extraction",
                trace_id=context.run_id,
            )
            enqueued.append("preference")

        execution_status = str(state.get("execution_status") or "completed")
        task_count = _task_count(state)
        substantive = is_substantive_progress(
            query=query,
            final_response=final_response,
            execution_status=execution_status,
        )
        decision = decide_post_turn_trigger(
            query=query,
            final_response=final_response,
            execution_status=execution_status,
            task_count=task_count,
            turn_count=1,
            uncompressed_tokens=sum(
                estimate_episodic_tokens(str(message.content or ""))
                for message in list(state.get("messages") or [])
            ),
        )
        if (
            not detect_important_state_change(query)
            and (decision.should_enqueue or substantive)
        ):
            await OutboxService.enqueue_episodic_extraction(
                run_id=context.run_id,
                user_id=int(context.user_id),
                tenant_id=str(context.tenant_id),
                conversation_id=(
                    int(context.conversation_id)
                    if context.conversation_id is not None
                    else None
                ),
                query=query,
                final_response=final_response,
                execution_status=execution_status,
                task_count=task_count,
                trigger_reasons=(
                    decision.reasons or ("progress_check",)
                ),
                forced=decision.forced,
                agent_id="orchestrator",
                task_id="episodic-extraction",
                trace_id=context.run_id,
            )
            enqueued.append("episodic")
    except Exception:
        # 记忆是回答后的增强能力，入队失败不能改变已生成的答案。
        logger.exception("post-turn memory enqueue failed run_id={}", context.run_id)
        return {
            "post_turn_memory_status": "enqueue_failed",
            "post_turn_memory_enqueued": enqueued,
        }

    return {
        "post_turn_memory_status": "enqueued" if enqueued else "not_needed",
        "post_turn_memory_enqueued": enqueued,
    }


__all__ = ["memory_recall_node", "post_turn_memory_node"]
