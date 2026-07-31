from types import SimpleNamespace

import pytest

from app.services.persistence import outbox_service


@pytest.mark.asyncio
async def test_outbox_extracts_implicit_preference_and_persists(monkeypatch):
    event = SimpleNamespace(
        id="event-1",
        event_type="memory.preference.extract",
        payload={
            "run_id": "run-1",
            "user_id": 7,
            "tenant_id": "tenant-1",
            "conversation_id": 12,
            "message_id": 34,
            "source_text": "我喜欢表格展示",
        },
    )
    preference = SimpleNamespace(
        memory_key="preferred_output_format",
        value="table",
        source="preference_rule",
        evidence="我喜欢表格展示",
        confidence=0.95,
    )
    created: list[dict] = []
    finished: list[tuple[str, bool]] = []

    async def fake_claim_one():
        return event

    async def fake_extract(_text):
        return preference

    async def fake_create(**kwargs):
        created.append(kwargs)

    async def fake_finish(event_id, *, success, error=""):
        finished.append((event_id, success))

    monkeypatch.setattr(outbox_service.OutboxService, "_claim_one", fake_claim_one)
    monkeypatch.setattr(outbox_service, "extract_preference", fake_extract)
    monkeypatch.setattr(outbox_service.MemoryService, "create", fake_create)
    monkeypatch.setattr(outbox_service.OutboxService, "_finish", fake_finish)

    assert await outbox_service.OutboxService.process_once() is True
    assert created[0]["memory_key"] == "preferred_output_format"
    assert created[0]["provenance"]["message_id"] == 34
    assert finished == [("event-1", True)]


@pytest.mark.asyncio
async def test_outbox_does_not_use_llm_for_sync_action(monkeypatch):
    event = SimpleNamespace(
        id="event-2",
        event_type="memory.preference.extract",
        payload={
            "run_id": "run-2",
            "user_id": 7,
            "tenant_id": "tenant-1",
            "source_text": "把回答语言改成英文",
        },
    )
    extraction_calls = 0

    async def fake_claim_one():
        return event

    async def fake_extract(_text):
        nonlocal extraction_calls
        extraction_calls += 1

    async def fake_finish(*_args, **_kwargs):
        return None

    monkeypatch.setattr(outbox_service.OutboxService, "_claim_one", fake_claim_one)
    monkeypatch.setattr(outbox_service, "extract_preference", fake_extract)
    monkeypatch.setattr(outbox_service.OutboxService, "_finish", fake_finish)

    assert await outbox_service.OutboxService.process_once() is True
    assert extraction_calls == 0
