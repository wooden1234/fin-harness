from __future__ import annotations

import pytest

from harness.agent.manager import AgentManager
from harness.llm.fake import FakeLlmAdapter
from harness.session.store import InMemorySessionStore
from harness.tools.runtime import ToolRuntime


@pytest.mark.asyncio
async def test_conversation_lookup_is_scoped_to_tenant_and_user() -> None:
    store = InMemorySessionStore()
    manager = AgentManager(store=store, llm=FakeLlmAdapter([]), runtime=ToolRuntime.builtin())

    first = await manager.session_for(tenant_id="tenant-a", user_id="user-a", conversation_id="42")
    same = await manager.session_for(tenant_id="tenant-a", user_id="user-a", conversation_id="42")
    other_user = await manager.session_for(tenant_id="tenant-a", user_id="user-b", conversation_id="42")
    other_tenant = await manager.session_for(tenant_id="tenant-b", user_id="user-a", conversation_id="42")

    assert same.session_id == first.session_id
    assert other_user.session_id != first.session_id
    assert other_tenant.session_id != first.session_id
