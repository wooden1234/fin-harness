import json

import pytest

from harness.session.store import InMemorySessionStore
from harness.session.types import EventDraft
from harness.tools.log_read import read_tool_log_definition


@pytest.mark.asyncio
async def test_read_tool_log_returns_a_slice_of_the_original_records() -> None:
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    records = [
        {"title": f"新闻{index}", "summary": "全文" * 300}
        for index in range(20)
    ]
    event = await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            surface_op="append",
            data={
                "call_id": "call-news",
                "name": "search_iwencai_news",
                "content": json.dumps({"data": {"data": records}}, ensure_ascii=False),
            },
        ),
    )
    read = read_tool_log_definition(store, header.session_id)
    result = await read.handler({"source_seq": event.seq, "offset": 4, "limit": 2})
    assert result["ok"] is True
    assert result["total"] == 20
    assert [item["title"] for item in result["items"]] == ["新闻4", "新闻5"]
    assert all(len(item["summary"]) <= 480 for item in result["items"])


@pytest.mark.asyncio
async def test_read_tool_log_caps_each_read_at_five_items() -> None:
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    records = [{"title": f"新闻{index}", "summary": "短"} for index in range(20)]
    event = await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            surface_op="append",
            data={
                "call_id": "call-news",
                "name": "search_iwencai_news",
                "content": json.dumps({"data": {"data": records}}),
            },
        ),
    )
    read = read_tool_log_definition(store, header.session_id)
    result = await read.handler({"source_seq": event.seq, "offset": 0, "limit": 20})
    assert len(result["items"]) == 5
