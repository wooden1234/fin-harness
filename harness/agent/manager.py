"""按 conversation 取得同一条 session 上的 Agent。"""

from __future__ import annotations

from typing import Any, Callable

from harness.agent.leases import InMemoryLeaseStore
from harness.agent.loop import Agent
from harness.llm.deepseek import DeepSeekAdapter
from harness.session.store import InMemorySessionStore, SessionStore
from harness.tools.runtime import ToolRuntime


class AgentManager:
    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        llm: Any | None = None,
        llm_factory: Callable[[], Any] | None = None,
        runtime: ToolRuntime | None = None,
        leases: Any | None = None,
    ) -> None:
        self.store = store or InMemorySessionStore()
        self._llm = llm
        self._llm_factory = llm_factory
        self._runtime = runtime or ToolRuntime.product()
        self._leases = leases or InMemoryLeaseStore()
        self._agents: dict[str, Agent] = {}

    def _llm_for(self) -> Any:
        if self._llm is not None:
            return self._llm
        if self._llm_factory is not None:
            return self._llm_factory()
        return DeepSeekAdapter()

    async def session_for(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | int | None,
    ):
        if conversation_id is not None:
            header = await self.store.find_by_conversation(conversation_id)
            if header is not None:
                if header.user_id != str(user_id) or header.tenant_id != str(tenant_id):
                    raise PermissionError("session_not_owned")
                return header
        return await self.store.create(
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            conversation_id=str(conversation_id) if conversation_id is not None else None,
        )

    async def get(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | int | None,
        owner_id: str | None = None,
    ) -> Agent:
        header = await self.session_for(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        agent = self._agents.get(header.session_id)
        if agent is None:
            agent = Agent(
                header.session_id,
                self.store,
                self._llm_for(),
                runtime=self._runtime,
                leases=self._leases,
                owner_id=owner_id or str(user_id),
            )
            self._agents[header.session_id] = agent
        return agent
