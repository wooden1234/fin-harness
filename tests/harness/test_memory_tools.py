from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.session.store import InMemorySessionStore
from harness.tools.memory import memory_tool_definitions


def _definitions(store, session_id: str):
    return {
        item.name: item
        for item in memory_tool_definitions(store, session_id, run_id="run-1")
    }


@pytest.mark.asyncio
async def test_memory_write_rejects_unknown_key():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="tenant-1", user_id="7")
    write = _definitions(store, header.session_id)["memory_write"]
    result = await write.handler({"memory_key": "not_a_key", "value": "zh-CN"})
    assert result["ok"] is False
    assert result["error"] == "unsupported_memory_key"


@pytest.mark.asyncio
async def test_memory_write_rejects_invalid_value(monkeypatch):
    store = InMemorySessionStore()
    header = await store.create(tenant_id="tenant-1", user_id="7")

    async def fail_create(**_kwargs):
        raise AssertionError("invalid value must not write")

    monkeypatch.setattr(
        "app.services.memory.memory_service.MemoryService.create",
        fail_create,
    )
    write = _definitions(store, header.session_id)["memory_write"]
    result = await write.handler({"memory_key": "response_language", "value": "fr-FR"})
    assert result["ok"] is False
    assert result["error"] == "invalid_memory_value"


@pytest.mark.asyncio
async def test_memory_write_passes_session_scope(monkeypatch):
    store = InMemorySessionStore()
    header = await store.create(tenant_id="tenant-1", user_id="7")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(version=2)

    monkeypatch.setattr(
        "app.services.memory.memory_service.MemoryService.create",
        fake_create,
    )
    write = _definitions(store, header.session_id)["memory_write"]
    result = await write.handler({"memory_key": "response_language", "value": "en-US"})
    assert result["ok"] is True
    assert result["memory_key"] == "response_language"
    assert result["value"] == "en-US"
    assert result["version"] == 2
    assert captured["tenant_id"] == "tenant-1"
    assert captured["user_id"] == 7
    assert captured["memory_key"] == "response_language"
    assert captured["value"] == "en-US"
    assert captured["actor_id"] == "7"


@pytest.mark.asyncio
async def test_memory_delete_passes_session_scope(monkeypatch):
    store = InMemorySessionStore()
    header = await store.create(tenant_id="tenant-1", user_id="7")
    captured: dict[str, object] = {}

    async def fake_revoke(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        "app.services.memory.memory_service.MemoryService.revoke_by_key",
        fake_revoke,
    )
    delete = _definitions(store, header.session_id)["memory_delete"]
    result = await delete.handler({"memory_key": "default_market"})
    assert result["ok"] is True
    assert result["revoked"] is True
    assert captured["tenant_id"] == "tenant-1"
    assert captured["user_id"] == 7
    assert captured["memory_key"] == "default_market"
