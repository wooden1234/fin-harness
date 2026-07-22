"""Agent 运行时租户上下文。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """由认证主体派生的不可变上下文，禁止使用请求体中的身份字段。"""

    tenant_id: str
    user_id: str
    conversation_id: str | None = None
    run_id: str | None = None

    @classmethod
    def from_user(
        cls,
        user: object,
        *,
        conversation_id: str | int | None = None,
        run_id: str | None = None,
    ) -> "AgentRuntimeContext":
        tenant_id = str(getattr(user, "tenant_id", "default"))
        user_id = str(getattr(user, "id"))
        if not tenant_id or tenant_id == "None":
            raise ValueError("认证主体缺少 tenant_id")
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            run_id=run_id,
        )
