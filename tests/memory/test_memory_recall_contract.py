from langchain_core.messages import HumanMessage, SystemMessage
import pytest
from types import SimpleNamespace

from agents.context import conversation_messages
from app.services.memory import memory_recall


def test_long_term_preference_is_before_current_turn_and_marks_override():
    messages = conversation_messages(
        {
            "memory_context": {"response_language": "zh-CN"},
            "messages": [HumanMessage(content="请用英文回答这一轮")],
        }
    )
    assert isinstance(messages[0], SystemMessage)
    assert "当前轮用户要求优先" in messages[0].content
    assert messages[-1].content == "请用英文回答这一轮"


def test_turn_preferences_are_separate_and_override_long_term_context():
    messages = conversation_messages(
        {
            "memory_context": {"response_language": "zh-CN"},
            "turn_preferences": {"response_language": "en-US"},
            "messages": [HumanMessage(content="这次请用英文回答")],
        }
    )

    assert "[用户长期偏好]" in messages[0].content
    assert "[本轮临时要求]" in messages[1].content
    assert "覆盖冲突的长期偏好" in messages[1].content


@pytest.mark.asyncio
async def test_preference_recall_uses_latest_sql_records_without_vector(monkeypatch):
    records = [
        SimpleNamespace(
            id="latest",
            tenant_id="tenant-1",
            user_id=7,
            status="active",
            expires_at=None,
            memory_key="response_language",
            value_json={"value": "en-US"},
            search_text="response_language en-US",
        ),
        SimpleNamespace(
            id="older-active",
            tenant_id="tenant-1",
            user_id=7,
            status="active",
            expires_at=None,
            memory_key="preferred_output_format",
            value_json={"value": "table"},
            search_text="preferred_output_format table",
        ),
    ]

    async def fake_list(**_kwargs):
        return records

    monkeypatch.setattr(memory_recall.MemoryService, "list", fake_list)

    recalled = await memory_recall.recall_preferences(
        tenant_id="tenant-1",
        user_id=7,
        query="回答语言",
        top_k=1,
        token_budget=100,
    )

    assert recalled == {"response_language": "en-US"}
