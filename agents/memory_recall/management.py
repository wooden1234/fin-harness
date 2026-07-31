"""长期记忆同步动作与含糊操作确认节点。"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState
from app.services.memory.memory_command import parse_memory_rule_action
from app.services.memory.memory_audit import MemoryAuditContext
from app.services.memory.memory_policy import validate_preference
from app.services.memory.memory_service import MemoryService


_CANCEL_MARKERS = ("取消", "算了", "不用了", "不操作了")
_CONFIRM_MARKERS = ("确认", "确定", "是的", "执行")
_KEY_LABELS = {
    "response_language": "回答语言",
    "response_detail_level": "回答详细程度",
    "preferred_output_format": "输出格式",
    "default_currency": "默认币种",
    "default_market": "默认市场",
    "default_compare_period": "比较周期",
    "citation_preference": "引用偏好",
}


def _latest_query(state: FinAgentState) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _candidate_payload(records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": record.id,
            "memory_key": record.memory_key,
            "value": (record.value_json or {}).get("value"),
            "version": record.version,
        }
        for record in records[:5]
    ]


def _candidate_prompt(action: str, candidates: list[dict[str, Any]]) -> str:
    operation = "删除" if action == "delete" else "修改"
    lines = [
        f"{index}. {_KEY_LABELS.get(item['memory_key'], item['memory_key'])}"
        f"（当前值：{item['value']}）"
        for index, item in enumerate(candidates, start=1)
    ]
    instruction = (
        f"请回复“确认{operation}第 N 个”或“取消”。"
        if action == "delete"
        else f"请回复“把第 N 个改成……”或“取消”。"
    )
    return f"你想{operation}哪条长期记忆？\n" + "\n".join(lines) + f"\n{instruction}"


def _selected_candidate(
    query: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    index_match = re.search(r"第\s*(\d+)\s*(?:个|条)", query)
    if index_match:
        index = int(index_match.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        return None
    for candidate in candidates:
        if candidate["memory_key"] in query:
            return candidate
        label = _KEY_LABELS.get(candidate["memory_key"], "")
        if label and label in query:
            return candidate
    if len(candidates) == 1 and any(marker in query for marker in _CONFIRM_MARKERS):
        return candidates[0]
    return None


async def _execute_resolved_action(
    *,
    action,
    context: AgentRuntimeContext,
    query: str,
) -> None:
    audit_context = MemoryAuditContext(
        tenant_id=str(context.tenant_id),
        user_id=int(context.user_id),
        agent_id="memory_management",
        task_id="memory_management",
        trace_id=context.run_id,
    )
    if action.kind in {"remember", "update"}:
        await MemoryService.create(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
            memory_key=action.memory_key,
            value=action.value,
            provenance={
                "source_type": f"{action.kind}_rule",
                "conversation_id": context.conversation_id,
                "run_id": context.run_id,
                "evidence": query[:500],
                "excerpt": query[:500],
            },
            actor_id=str(context.user_id),
            confidence=1.0,
            audit_context=audit_context,
        )
    elif action.kind == "delete":
        await MemoryService.revoke_by_key(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
            memory_key=action.memory_key,
            actor_id=str(context.user_id),
            audit_context=audit_context,
        )


async def memory_action_node(
    state: FinAgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, Any]:
    """在输入护栏之后执行同步动作，含糊操作先保存候选等待确认。"""
    context = runtime.context
    if context is None:
        return {"memory_action_handled": False}

    query = _latest_query(state)
    action = parse_memory_rule_action(query)
    pending = dict(state.get("pending_memory_action") or {})

    if pending and any(marker in query for marker in _CANCEL_MARKERS):
        return {
            "pending_memory_action": {},
            "memory_action_handled": True,
            "summary": "已取消本次长期记忆操作。",
            "route": "memory_management",
        }

    candidates = list(pending.get("candidates") or [])
    selected = _selected_candidate(query, candidates) if pending else None
    if selected is not None and pending.get("action") == "delete":
        if "删除" not in query and not any(marker in query for marker in _CONFIRM_MARKERS):
            return {"memory_action_handled": False}
        deleted = await MemoryService.revoke(
            tenant_id=context.tenant_id,
            user_id=int(context.user_id),
            memory_id=str(selected["id"]),
            actor_id=str(context.user_id),
            audit_context=MemoryAuditContext(
                tenant_id=str(context.tenant_id),
                user_id=int(context.user_id),
                agent_id="memory_management",
                task_id="memory_management",
                trace_id=context.run_id,
            ),
        )
        return {
            "pending_memory_action": {},
            "memory_action_handled": True,
            "memory_cache_bypass": True,
            "summary": "已删除选中的长期记忆。" if deleted else "该记忆已不存在或已经失效。",
            "route": "memory_management",
        }

    if selected is not None and pending.get("action") == "update":
        if action.kind != "update" or action.value is None:
            return {
                "memory_action_handled": True,
                "summary": "请同时说明要修改成什么值，例如“把第 2 个改成英文”。",
                "route": "memory_management",
            }
        try:
            validate_preference(str(selected["memory_key"]), action.value)
        except ValueError:
            return {
                "memory_action_handled": True,
                "summary": "新的值不适用于所选记忆，请重新选择或换一个值。",
                "route": "memory_management",
            }
        try:
            updated = await MemoryService.update(
                tenant_id=context.tenant_id,
                user_id=int(context.user_id),
                memory_id=str(selected["id"]),
                value=action.value,
                expected_version=int(selected["version"]),
                actor_id=str(context.user_id),
                reason="用户确认含糊修改候选",
                audit_context=MemoryAuditContext(
                    tenant_id=str(context.tenant_id),
                    user_id=int(context.user_id),
                    agent_id="memory_management",
                    task_id="memory_management",
                    trace_id=context.run_id,
                ),
            )
        except ValueError:
            updated = None
        return {
            "pending_memory_action": {},
            "memory_action_handled": True,
            "memory_cache_bypass": True,
            "summary": "已修改选中的长期记忆。" if updated else "该记忆已不存在或已经变化。",
            "route": "memory_management",
        }

    if action.kind in {"remember", "update", "delete"} and action.resolved:
        await _execute_resolved_action(action=action, context=context, query=query)
        return {
            "pending_memory_action": {},
            "memory_action_handled": False,
            "memory_cache_bypass": True,
        }

    if action.kind not in {"update", "delete"}:
        return {"memory_action_handled": False}

    records = await MemoryService.list(
        tenant_id=context.tenant_id,
        user_id=int(context.user_id),
    )
    if action.memory_key is not None:
        records = [
            record for record in records if record.memory_key == action.memory_key
        ]
    candidates = _candidate_payload(records)
    if not candidates:
        return {
            "pending_memory_action": {},
            "memory_action_handled": True,
            "summary": "没有找到可供管理的长期记忆。",
            "route": "memory_management",
        }
    return {
        "pending_memory_action": {
            "action": action.kind,
            "candidates": candidates,
        },
        "memory_action_handled": True,
        "summary": _candidate_prompt(action.kind, candidates),
        "route": "memory_management",
    }


def memory_action_edge(state: FinAgentState) -> str:
    return "final_answer" if state.get("memory_action_handled") else "continue"


__all__ = ["memory_action_edge", "memory_action_node"]
