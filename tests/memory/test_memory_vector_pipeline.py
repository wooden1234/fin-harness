from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agents.memory_recall import planning
from app.services.memory import memory_index_service, memory_loader, memory_store


def _record(
    memory_id: str,
    *,
    tenant_id: str = "tenant-1",
    user_id: int = 7,
    status: str = "active",
    expires_at=None,
    value: str = "own-history",
):
    return SimpleNamespace(
        id=memory_id,
        tenant_id=tenant_id,
        user_id=user_id,
        memory_type="episodic",
        memory_key=f"event:{memory_id}",
        value_json={"summary": value},
        display_text=value,
        version=1,
        status=status,
        expires_at=expires_at,
    )


@pytest.mark.asyncio
async def test_preference_cannot_write_vector_index():
    with pytest.raises(ValueError, match="memory_type_not_vectorizable"):
        await memory_store.upsert_memory_index(
            memory_id="memory-1",
            tenant_id="tenant-1",
            user_id=7,
            memory_type="preference",
            version=1,
            search_text="response_language zh-CN",
        )


@pytest.mark.asyncio
async def test_preference_outbox_event_is_rejected_before_vector_write(
    monkeypatch,
):
    record = SimpleNamespace(
        id="memory-1",
        tenant_id="tenant-1",
        user_id=7,
        memory_type="preference",
        memory_key="response_language",
        value_json={"value": "zh-CN"},
        search_text="response_language zh-CN",
        version=1,
        status="active",
    )
    vector_writes = 0

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return record

    async def fake_upsert(**_kwargs):
        nonlocal vector_writes
        vector_writes += 1

    monkeypatch.setattr(
        memory_index_service,
        "AsyncSessionLocal",
        FakeSession,
    )
    monkeypatch.setattr(
        memory_index_service,
        "upsert_memory_index",
        fake_upsert,
    )

    await memory_index_service.MemoryIndexService.upsert_from_event(
        {
            "memory_id": "memory-1",
            "tenant_id": "tenant-1",
            "user_id": 7,
            "memory_type": "preference",
            "version": 1,
        }
    )

    assert vector_writes == 0


@pytest.mark.asyncio
async def test_vector_store_search_exposes_ids_only(monkeypatch):
    class FakeStore:
        async def asearch(self, *_args, **_kwargs):
            return [
                SimpleNamespace(key="memory-1", value={"secret": "ignored"}),
                SimpleNamespace(key="memory-1", value={"secret": "ignored"}),
                SimpleNamespace(key="memory-2", value={"secret": "ignored"}),
            ]

    monkeypatch.setattr(memory_store, "get_memory_store", lambda: FakeStore())

    result = await memory_store.search_memory_ids(
        tenant_id="tenant-1",
        user_id=7,
        memory_type="episodic",
        query="之前的讨论",
        limit=5,
    )

    assert result == ["memory-1", "memory-2"]


@pytest.mark.asyncio
async def test_vector_index_payload_does_not_store_memory_value(
    monkeypatch,
):
    captured: dict = {}

    class FakeStore:
        async def aput(self, namespace, key, value):
            captured.update(
                namespace=namespace,
                key=key,
                value=value,
            )

    monkeypatch.setattr(memory_store, "get_memory_store", lambda: FakeStore())

    await memory_store.upsert_memory_index(
        memory_id="event-1",
        tenant_id="tenant-1",
        user_id=7,
        memory_type="episodic",
        search_text="上次讨论了资产配置",
        version=1,
    )

    assert captured["key"] == "event-1"
    assert captured["value"] == {
        "memory_id": "event-1",
        "version": 1,
        "search_text": "上次讨论了资产配置",
    }


@pytest.mark.asyncio
async def test_forged_cross_tenant_vector_id_never_enters_agent_state(
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    records = [
        _record("own"),
        _record("expired", expires_at=now - timedelta(seconds=1)),
        _record("revoked", status="revoked"),
    ]
    metrics: dict[str, int] = {}

    async def fake_search_memory_ids(**_kwargs):
        return ["cross", "own", "expired", "missing", "revoked"]

    async def fake_verify(**kwargs):
        assert kwargs["tenant_id"] == "tenant-1"
        assert kwargs["user_id"] == 7
        assert kwargs["memory_type"] == "episodic"
        assert kwargs["memory_ids"] == (
            "cross",
            "own",
            "expired",
            "missing",
            "revoked",
        )
        return records, ("cross",)

    async def fake_load_for_agent(**_kwargs):
        return SimpleNamespace(as_dict=lambda: {})

    def fake_increment(name: str, value: int = 1):
        metrics[name] = metrics.get(name, 0) + value

    monkeypatch.setattr(
        memory_loader,
        "search_memory_ids",
        fake_search_memory_ids,
    )
    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_vector_candidates_for_verification",
        fake_verify,
    )
    monkeypatch.setattr(memory_loader, "increment", fake_increment)
    monkeypatch.setattr(
        planning.MemoryLoader,
        "load_for_agent",
        fake_load_for_agent,
    )

    result = await planning.load_task_memories_node(
        {
            "memory_requirements": {
                "general": {
                    "agent_id": "general_agent",
                    "memory_keys": ["response_language"],
                    "semantic_memory_type": "episodic",
                    "semantic_query": "结合我之前的讨论",
                    "semantic_top_k": 5,
                }
            }
        },
        SimpleNamespace(
            context=SimpleNamespace(tenant_id="tenant-1", user_id=7)
        ),
    )

    context = result["task_memory_context"]["general"]
    assert [item["memory_id"] for item in context["semantic_history"]] == [
        "own"
    ]
    assert metrics["memory_vector_cross_tenant_rejected_total"] == 1
    assert metrics["memory_vector_expired_rejected_total"] == 1
    assert metrics["memory_vector_stale_rejected_total"] == 2


@pytest.mark.asyncio
async def test_semantic_memory_type_is_agent_whitelisted():
    with pytest.raises(
        PermissionError,
        match="agent_semantic_memory_type_denied",
    ):
        await memory_loader.MemoryLoader.load_semantic_for_agent(
            tenant_id="tenant-1",
            user_id=7,
            agent_id="general_agent",
            memory_type="preference",
            query="历史偏好",
            top_k=5,
        )
