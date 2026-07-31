from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from app.services.memory import memory_loader


@pytest.mark.asyncio
async def test_loader_queries_exact_keys_in_trusted_scope(monkeypatch):
    captured: dict = {}

    async def fake_list_by_keys(**kwargs):
        captured.update(kwargs)
        return [
            SimpleNamespace(
                memory_key="response_language",
                memory_type="preference",
                value_json={"value": "zh-CN"},
                version=2,
                expires_at=None,
            )
        ]

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )
    projection = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=("response_language", "response_language"),
    )

    assert captured == {
        "tenant_id": "tenant-1",
        "user_id": 7,
        "memory_type": "preference",
        "memory_keys": ("response_language",),
    }
    assert projection.as_dict() == {"response_language": "zh-CN"}
    assert projection.keys == ("response_language",)


@pytest.mark.asyncio
async def test_projection_returns_defensive_value_copies(monkeypatch):
    async def fake_list_by_keys(**_kwargs):
        return [
            SimpleNamespace(
                memory_key="preferred_output_format",
                memory_type="preference",
                value_json={"value": {"format": "table"}},
                version=1,
                expires_at=None,
            )
        ]

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )
    projection = await memory_loader.MemoryLoader.load_by_keys(
        tenant_id="tenant-1",
        user_id=7,
        memory_keys=("preferred_output_format",),
    )

    first = projection.as_dict()
    first["preferred_output_format"]["format"] = "markdown"

    assert projection.as_dict() == {
        "preferred_output_format": {"format": "table"}
    }
    with pytest.raises(FrozenInstanceError):
        projection.user_id = 8


@pytest.mark.asyncio
async def test_loader_rejects_untrusted_scope_before_query(monkeypatch):
    query_called = False

    async def fake_list_by_keys(**_kwargs):
        nonlocal query_called
        query_called = True
        return []

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )

    with pytest.raises(ValueError, match="trusted_tenant_id"):
        await memory_loader.MemoryLoader.load_by_keys(
            tenant_id="",
            user_id=7,
            memory_keys=("response_language",),
        )
    with pytest.raises(ValueError, match="trusted_user_id"):
        await memory_loader.MemoryLoader.load_by_keys(
            tenant_id="tenant-1",
            user_id=0,
            memory_keys=("response_language",),
        )
    assert query_called is False


@pytest.mark.asyncio
async def test_loader_enforces_agent_memory_whitelist(monkeypatch):
    query_called = False

    async def fake_list_by_keys(**_kwargs):
        nonlocal query_called
        query_called = True
        return []

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )

    with pytest.raises(PermissionError, match="agent_memory_key_denied"):
        await memory_loader.MemoryLoader.load_for_agent(
            tenant_id="tenant-1",
            user_id=7,
            agent_id="general_agent",
            memory_keys=("default_market",),
        )
    assert query_called is False


@pytest.mark.asyncio
async def test_loader_rejects_forbidden_memory_key(monkeypatch):
    query_called = False

    async def fake_list_by_keys(**_kwargs):
        nonlocal query_called
        query_called = True
        return []

    monkeypatch.setattr(
        memory_loader.MemoryService,
        "list_by_keys",
        fake_list_by_keys,
    )

    with pytest.raises(ValueError, match="memory_key_type_mismatch"):
        await memory_loader.MemoryLoader.load_by_keys(
            tenant_id="tenant-1",
            user_id=7,
            memory_keys=("credential_secret",),
        )
    assert query_called is False
