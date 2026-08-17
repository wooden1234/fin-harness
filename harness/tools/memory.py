"""显式长期偏好写删；身份从 session header 闭包绑定。"""

from __future__ import annotations

from typing import Any

from harness.memory_specs import get_agent_spec
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result

_AGENT_ID = "fin_agent"
_MEMORY_KEYS = get_agent_spec(_AGENT_ID).memory_keys

_WRITE_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memory_key": {
            "type": "string",
            "enum": list(_MEMORY_KEYS),
            "description": "白名单偏好 key",
        },
        "value": {"type": "string", "description": "该 key 允许的枚举值"},
    },
    "required": ["memory_key", "value"],
}

_DELETE_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memory_key": {
            "type": "string",
            "enum": list(_MEMORY_KEYS),
            "description": "要忘记的白名单偏好 key",
        },
    },
    "required": ["memory_key"],
}


async def _session_scope(store: Any, session_id: str) -> tuple[str, int] | str:
    try:
        header = await store.get(session_id)
    except Exception:  # noqa: BLE001
        return "memory_scope_unavailable"
    try:
        tenant_id = str(getattr(header, "tenant_id", "") or "").strip()
        user_id = int(getattr(header, "user_id", 0))
    except (TypeError, ValueError):
        return "memory_scope_unavailable"
    if not tenant_id or tenant_id == "None" or user_id <= 0:
        return "memory_scope_unavailable"
    return tenant_id, user_id


def _normalized_key(raw: Any) -> str | None:
    memory_key = str(raw or "").strip()
    if memory_key not in _MEMORY_KEYS:
        return None
    return memory_key


def memory_tool_definitions(
    store: Any,
    session_id: str,
    *,
    run_id: str,
) -> tuple[ToolDefinition, ToolDefinition]:
    async def _write(arguments: dict[str, Any]) -> dict[str, Any]:
        memory_key = _normalized_key(arguments.get("memory_key"))
        if memory_key is None:
            return error_result("unsupported_memory_key")
        value = arguments.get("value")
        if not isinstance(value, str) or not value.strip():
            return error_result("invalid_memory_value")
        value = value.strip()
        scope = await _session_scope(store, session_id)
        if isinstance(scope, str):
            return error_result(scope)
        tenant_id, user_id = scope
        try:
            from app.services.memory.memory_audit import MemoryAuditContext
            from app.services.memory.memory_policy import validate_preference
            from app.services.memory.memory_service import MemoryService

            validate_preference(memory_key, value)
            record = await MemoryService.create(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_key=memory_key,
                value=value,
                actor_id=str(user_id),
                audit_context=MemoryAuditContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    agent_id=_AGENT_ID,
                    task_id="memory_write",
                    trace_id=run_id,
                ),
            )
        except ValueError:
            return error_result("invalid_memory_value")
        except Exception:  # noqa: BLE001
            return error_result("memory_write_failed")
        return {
            "ok": True,
            "memory_key": memory_key,
            "value": value,
            "version": int(getattr(record, "version", 1) or 1),
            "content": f"已记住 {memory_key}={value}",
        }

    async def _delete(arguments: dict[str, Any]) -> dict[str, Any]:
        memory_key = _normalized_key(arguments.get("memory_key"))
        if memory_key is None:
            return error_result("unsupported_memory_key")
        scope = await _session_scope(store, session_id)
        if isinstance(scope, str):
            return error_result(scope)
        tenant_id, user_id = scope
        try:
            from app.services.memory.memory_audit import MemoryAuditContext
            from app.services.memory.memory_service import MemoryService

            revoked = await MemoryService.revoke_by_key(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_key=memory_key,
                actor_id=str(user_id),
                audit_context=MemoryAuditContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    agent_id=_AGENT_ID,
                    task_id="memory_delete",
                    trace_id=run_id,
                ),
            )
        except Exception:  # noqa: BLE001
            return error_result("memory_delete_failed")
        return {
            "ok": True,
            "memory_key": memory_key,
            "revoked": bool(revoked),
            "content": (
                f"已忘记 {memory_key}" if revoked else f"没有可删除的 {memory_key}"
            ),
        }

    write = ToolDefinition(
        tool_id="memory_write",
        name="memory_write",
        description=(
            "写入或更新一项跨会话回答偏好。仅在用户明确说请记住、以后默认或改成时调用。"
        ),
        handler=_write,
        openai_schema=function_schema(
            "memory_write",
            "记住或更新长期回答偏好",
            _WRITE_PARAMETERS,
        ),
        is_concurrency_safe=False,
        read_only=False,
        timeout_seconds=5.0,
    )
    delete = ToolDefinition(
        tool_id="memory_delete",
        name="memory_delete",
        description="删除一项跨会话回答偏好。仅在用户明确说忘记或不再记住时调用。",
        handler=_delete,
        openai_schema=function_schema(
            "memory_delete",
            "忘记一项长期回答偏好",
            _DELETE_PARAMETERS,
        ),
        is_concurrency_safe=False,
        read_only=False,
        timeout_seconds=5.0,
    )
    return write, delete
