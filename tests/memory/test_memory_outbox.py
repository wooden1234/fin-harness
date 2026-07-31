from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from app.services.persistence import outbox_service
from app.services.memory.memory_episodic_extraction import ExtractedEpisodicMemory


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


@pytest.mark.asyncio
async def test_outbox_extracts_completed_task_into_episodic_memory(monkeypatch):
    now = datetime.now(timezone.utc)
    messages = [
        {
            "id": 10,
            "sender": "user",
            "content": "分析贵州茅台现金流",
            "created_at": (now - timedelta(minutes=2)).isoformat(),
        },
        {
            "id": 11,
            "sender": "assistant",
            "content": "已完成现金流和利润质量分析。" * 20,
            "created_at": now.isoformat(),
        },
    ]
    created: list[dict] = []

    async def fake_latest(**_kwargs):
        return None

    async def fake_messages(*_args, **_kwargs):
        return messages

    async def fake_extract(*_args, **_kwargs):
        return ExtractedEpisodicMemory(
            event_type="task_result",
            subject_key="stock_analysis",
            topic="贵州茅台分析",
            summary="完成贵州茅台现金流和利润质量分析。",
            facts=("分析了现金流",),
            conclusion="后续继续跟踪利润质量。",
            evidence=("分析贵州茅台现金流",),
            confidence=0.9,
            quality_score=0.95,
        )

    async def fake_duplicate(**_kwargs):
        return None

    async def fake_create(**kwargs):
        created.append(kwargs)

    monkeypatch.setattr(
        outbox_service.MemoryService,
        "latest_episodic_for_conversation",
        fake_latest,
    )
    monkeypatch.setattr(
        outbox_service.ConversationService,
        "get_conversation_messages",
        fake_messages,
    )
    monkeypatch.setattr(outbox_service, "extract_episodic_memory", fake_extract)
    monkeypatch.setattr(
        outbox_service.MemoryService,
        "find_duplicate_episodic",
        fake_duplicate,
    )
    monkeypatch.setattr(
        outbox_service.MemoryService,
        "create_episodic",
        fake_create,
    )

    await outbox_service.OutboxService._process_episodic_extraction(
        {
            "run_id": "run-episodic",
            "tenant_id": "tenant-1",
            "user_id": 7,
            "conversation_id": 12,
            "message_id": 11,
            "query": "分析贵州茅台现金流",
            "final_response": "已完成现金流和利润质量分析。" * 20,
            "execution_status": "completed",
            "task_count": 1,
            "trigger_reasons": ["task_completed"],
            "forced": False,
            "agent_id": "orchestrator",
            "task_id": "episodic-extraction",
            "trace_id": "run-episodic",
        }
    )

    assert created[0]["event_key"] == "task_result:12:11"
    assert created[0]["source_message_id"] == 11
    assert created[0]["value"]["subject_key"] == "stock_analysis"
