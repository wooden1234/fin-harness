from types import SimpleNamespace

from langchain_core.messages import HumanMessage
import pytest

from agents.memory_recall import management
from app.services.memory.memory_episodic_extraction import ExtractedEpisodicMemory


def _runtime():
    return SimpleNamespace(
        context=SimpleNamespace(
            tenant_id="tenant-1",
            user_id=7,
            conversation_id=12,
            run_id="run-1",
        )
    )


@pytest.mark.asyncio
async def test_ambiguous_delete_returns_scoped_candidates(monkeypatch):
    records = [
        SimpleNamespace(
            id="memory-1",
            memory_key="response_language",
            value_json={"value": "zh-CN"},
            version=1,
        ),
        SimpleNamespace(
            id="memory-2",
            memory_key="preferred_output_format",
            value_json={"value": "table"},
            version=2,
        ),
    ]

    async def fake_list(**_kwargs):
        return records

    monkeypatch.setattr(management.MemoryService, "list", fake_list)
    result = await management.memory_action_node(
        {"messages": [HumanMessage(content="删除之前那个记忆")]},
        _runtime(),
    )

    assert result["memory_action_handled"] is True
    assert result["pending_memory_action"]["action"] == "delete"
    assert len(result["pending_memory_action"]["candidates"]) == 2
    assert "确认删除第 N 个" in result["summary"]


@pytest.mark.asyncio
async def test_confirmed_candidate_delete_executes_synchronously(monkeypatch):
    revoked: list[str] = []

    async def fake_revoke(**kwargs):
        revoked.append(kwargs["memory_id"])
        return True

    monkeypatch.setattr(management.MemoryService, "revoke", fake_revoke)
    result = await management.memory_action_node(
        {
            "messages": [HumanMessage(content="确认删除第2个")],
            "pending_memory_action": {
                "action": "delete",
                "candidates": [
                    {
                        "id": "memory-1",
                        "memory_key": "response_language",
                        "value": "zh-CN",
                        "version": 1,
                    },
                    {
                        "id": "memory-2",
                        "memory_key": "preferred_output_format",
                        "value": "table",
                        "version": 2,
                    },
                ],
            },
        },
        _runtime(),
    )

    assert revoked == ["memory-2"]
    assert result["pending_memory_action"] == {}
    assert result["memory_cache_bypass"] is True
    assert result["summary"] == "已删除选中的长期记忆。"


@pytest.mark.asyncio
async def test_explicit_state_change_is_persisted_before_answer(monkeypatch):
    created: list[dict] = []

    async def fake_extract(*_args, **_kwargs):
        return ExtractedEpisodicMemory(
            event_type="state_change",
            subject_key="retrieval_strategy",
            topic="检索策略变化",
            summary="检索策略调整为混合检索。",
            facts=("旧策略为纯向量检索", "新策略为混合检索"),
            conclusion="后续使用混合检索。",
            evidence=("不再使用纯向量检索，改成混合检索",),
            confidence=0.95,
            quality_score=0.99,
        )

    async def fake_create(**kwargs):
        created.append(kwargs)

    monkeypatch.setattr(management, "extract_episodic_memory", fake_extract)
    monkeypatch.setattr(
        management.MemoryService,
        "create_episodic",
        fake_create,
    )

    result = await management.memory_action_node(
        {
            "messages": [
                HumanMessage(content="不再使用纯向量检索，改成混合检索。")
            ]
        },
        _runtime(),
    )

    assert result["memory_action_handled"] is False
    assert created[0]["event_key"].startswith("state_change:retrieval_strategy")
    assert created[0]["value"]["event_type"] == "state_change"
