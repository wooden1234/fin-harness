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


@pytest.mark.asyncio
async def test_read_tool_log_slices_plain_text_by_character() -> None:
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    gold = "1174464377.35"
    body = "甲" * 5000 + gold + "乙" * 5000
    event = await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            surface_op="append",
            data={"call_id": "call-text", "name": "web.search", "ok": True, "content": body},
        ),
    )
    read = read_tool_log_definition(store, header.session_id)
    missed = await read.handler({"source_seq": event.seq, "offset": 0, "limit": 1})
    assert missed["ok"] is False
    found = await read.handler({"source_seq": event.seq, "char_offset": 4096, "char_limit": 2000})
    assert found["ok"] is True
    assert found["chars"] == len(body)
    assert gold in found["content"]
    assert found["next_offset"] == 4096 + 2000
    capped = await read.handler({"source_seq": event.seq, "char_offset": 0, "char_limit": 9000})
    assert len(capped["content"]) == 2000


@pytest.mark.asyncio
async def test_plain_text_trim_pointer_reads_back_the_omitted_span() -> None:
    from harness.compaction.compact import maybe_compact

    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    gold = "1174464377.35"
    body = "甲" * 5000 + gold + "乙" * 5000
    await store.append(header.session_id, EventDraft(event_type="turn/start", turn=1, data={"turn": 1}))
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "查数字", "source": "user"},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="assistant/message",
            turn=1,
            surface_op="append",
            data={"content": "", "tool_calls": [{"call_id": "c1", "name": "web.search", "arguments": "{}"}]},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="tool/result",
            turn=1,
            surface_op="append",
            data={"call_id": "c1", "name": "web.search", "ok": True, "content": body},
        ),
    )
    changed = await maybe_compact(
        store=store,
        session_id=header.session_id,
        turn=1,
        run_id="r",
        token_limit=30,
        allow_llm=False,
    )
    events = await store.load_events(header.session_id)
    replace = next(event for event in events if event.surface_op == "replace")
    visible = str(replace.data["content"])
    assert changed is True
    assert gold not in visible
    ref = json.loads(visible.splitlines()[-1])["log_ref"]
    start, _end = ref["span"]
    read = read_tool_log_definition(store, header.session_id)
    found = await read.handler({"source_seq": ref["seq"], "char_offset": start, "char_limit": 2000})
    assert gold in found["content"]
