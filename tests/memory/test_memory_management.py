from types import SimpleNamespace

from langchain_core.messages import HumanMessage
import pytest

from agents.memory_recall import management


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
