from __future__ import annotations

import pytest

from harness.compaction.compact import (
    current_turn_todos,
    drop_oldest_turn_facts,
    latest_committed_summary,
    load_summary_v2,
    maybe_compact,
    merge_summary,
    merge_summary_v2,
    prune_facts,
    shrink_summary_v2,
)
from harness.compaction.meter import request_tokens
from harness.compaction.policy import CompactPolicy
from harness.compaction.schema import (
    CompactionDelta,
    CompactionDeltaV2,
    CompactionFact,
    CompactionItemV2,
    CompactionSummary,
    CompactionSummaryV2,
    render_summary_text,
)
from harness.contracts.errors import ContextBudgetExhaustedError
from harness.llm.fake import FakeLlmAdapter
from harness.session.store import InMemorySessionStore
from harness.session.surface import derive_messages
from harness.session.types import EventDraft


def _delta(**kwargs: object) -> CompactionDelta:
    return CompactionDelta.model_validate(kwargs)


def _contents(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("content") or "") for row in rows]


async def _seed_turn(store: InMemorySessionStore, session_id: str, turn: int, *, user: str, tool: str) -> None:
    await store.append(session_id, EventDraft(event_type="turn/start", turn=turn, data={"turn": turn}))
    await store.append(
        session_id,
        EventDraft(
            event_type="user/message",
            turn=turn,
            surface_op="append",
            data={"content": user, "source": "user"},
        ),
    )
    await store.append(
        session_id,
        EventDraft(
            event_type="assistant/message",
            turn=turn,
            surface_op="append",
            data={
                "content": "",
                "tool_calls": [{"call_id": f"c{turn}", "name": "web.search", "arguments": "{}"}],
            },
        ),
    )
    await store.append(
        session_id,
        EventDraft(
            event_type="tool/result",
            turn=turn,
            surface_op="append",
            data={"call_id": f"c{turn}", "name": "web.search", "ok": True, "content": tool},
        ),
    )


@pytest.mark.asyncio
async def test_structured_summary_writes_facts_and_turn_progress():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await _seed_turn(store, header.session_id, 1, user="甲" * 40, tool="乙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "当前问题", "source": "user"},
        ),
    )
    llm = FakeLlmAdapter(
        structured_deltas=[
            _delta(
                new_facts=[
                    {
                        "company": "乙公司",
                        "period": "2024",
                        "metric": "营收",
                        "value": "100",
                        "unit": "亿元",
                        "evidence_id": "ev-1",
                        "turn": 1,
                    }
                ],
                new_completed_items=["已查乙公司营收"],
                new_open_items=["净利润未查"],
            )
        ]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=llm,
        turn=1,
        run_id="r",
        token_limit=20,
        allow_llm=True,
        policy=CompactPolicy(context_window=40, retain_ratio=0.2, prune_chars=10_000),
    )
    events = await store.load_events(header.session_id)
    summary = next(event for event in events if event.event_type == "compaction/summary")
    structured = summary.data["structured"]
    assert structured["facts"][0]["evidence_id"] == "ev-1"
    assert "已查乙公司营收" in _contents(structured["completed_items"])
    assert "净利润未查" in _contents(structured["open_items"])
    assert summary.data["compaction_schema_version"] == 2
    messages = derive_messages(events)
    assert messages[-1].content == "当前问题"
    assert any(item.source == "compaction" for item in messages)
    assert "已确认事实" in summary.data["content"]


@pytest.mark.asyncio
async def test_second_compaction_folds_previous_summary():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await _seed_turn(store, header.session_id, 1, user="甲" * 40, tool="乙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "追问一", "source": "user"},
        ),
    )
    first = FakeLlmAdapter(
        structured_deltas=[
            _delta(
                new_facts=[{"company": "乙", "metric": "营收", "value": "1", "turn": 1}],
                new_open_items=["净利润未查"],
            )
        ]
    )
    policy = CompactPolicy(context_window=40, retain_ratio=0.2, prune_chars=10_000)
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=first,
        turn=1,
        run_id="r1",
        token_limit=20,
        allow_llm=True,
        policy=policy,
    )
    first_summary = next(
        event
        for event in await store.load_events(header.session_id)
        if event.event_type == "compaction/summary"
    )
    await _seed_turn(store, header.session_id, 1, user="更多" * 20, tool="丙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "追问二", "source": "user"},
        ),
    )
    second = FakeLlmAdapter(
        structured_deltas=[
            _delta(
                new_facts=[{"company": "丙", "metric": "营收", "value": "2", "turn": 1}],
                new_completed_items=["净利润未查"],
                new_open_items=["估值未查"],
            )
        ]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=second,
        turn=1,
        run_id="r2",
        token_limit=20,
        allow_llm=True,
        policy=policy,
    )
    events = await store.load_events(header.session_id)
    summaries = [event for event in events if event.event_type == "compaction/summary"]
    assert len(summaries) == 2
    assert first_summary.seq in set(summaries[1].source_event_seqs)
    structured_prompts = [
        item["prompt"] for item in second.requests if item.get("kind") == "complete_structured"
    ]
    assert structured_prompts
    assert "[压缩摘要]" not in structured_prompts[0]
    messages = derive_messages(events)
    compacted = [item for item in messages if item.source == "compaction"]
    assert len(compacted) == 1
    structured = summaries[1].data["structured"]
    assert "净利润未查" not in _contents(structured["open_items"])
    assert "估值未查" in _contents(structured["open_items"])


@pytest.mark.asyncio
async def test_incomplete_pair_stays_out_of_summary_prefix():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "甲" * 40, "source": "user"},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "当前问题", "source": "user"},
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="assistant/message",
            turn=1,
            surface_op="append",
            data={
                "content": "",
                "tool_calls": [{"call_id": "open-1", "name": "web.search", "arguments": "{}"}],
            },
        ),
    )
    llm = FakeLlmAdapter(
        structured_deltas=[_delta(new_facts=[{"note": "较早问题", "turn": 1}])]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=llm,
        turn=1,
        run_id="r",
        token_limit=10,
        allow_llm=True,
        policy=CompactPolicy(context_window=40, retain_ratio=0.05, prune_chars=10_000),
    )
    events = await store.load_events(header.session_id)
    summary = next(event for event in events if event.event_type == "compaction/summary")
    assistant = next(event for event in events if event.event_type == "assistant/message")
    current = next(
        event
        for event in events
        if event.event_type == "user/message" and event.data.get("content") == "当前问题"
    )
    dropped = set(summary.source_event_seqs)
    assert assistant.seq not in dropped
    assert current.seq not in dropped


@pytest.mark.asyncio
async def test_todo_write_overrides_task_lists():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await _seed_turn(store, header.session_id, 1, user="甲" * 40, tool="乙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="todo/write",
            turn=1,
            data={
                "todos": [
                    {"content": "查营收", "status": "completed"},
                    {"content": "查净利润", "status": "pending"},
                ]
            },
        ),
    )
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "当前问题", "source": "user"},
        ),
    )
    llm = FakeLlmAdapter(
        structured_deltas=[
            _delta(
                new_open_items=["LLM 多写的遗漏"],
                new_completed_items=["不该覆盖 todo 已完成"],
            )
        ]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=llm,
        turn=1,
        run_id="r",
        token_limit=20,
        allow_llm=True,
        policy=CompactPolicy(context_window=40, retain_ratio=0.2, prune_chars=10_000),
    )
    events = await store.load_events(header.session_id)
    structured = next(event for event in events if event.event_type == "compaction/summary").data[
        "structured"
    ]
    assert _contents(structured["completed_items"])[0] == "查营收"
    assert "查净利润" in _contents(structured["open_items"])
    assert "LLM 多写的遗漏" in _contents(structured["open_items"])
    todos = current_turn_todos(events, turn=1)
    assert len(todos) == 2


@pytest.mark.asyncio
async def test_prior_turn_facts_kept_without_open_items():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await _seed_turn(store, header.session_id, 1, user="甲" * 40, tool="茅台营收1740")
    await store.append(
        header.session_id,
        EventDraft(event_type="turn/end", turn=1, data={"turn": 1, "reason": "completed"}),
    )
    await store.append(header.session_id, EventDraft(event_type="turn/start", turn=2, data={"turn": 2}))
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=2,
            surface_op="append",
            data={"content": "对比五粮液" + "问" * 20, "source": "user"},
        ),
    )
    llm = FakeLlmAdapter(
        structured_deltas=[
            _delta(
                new_facts=[
                    {
                        "company": "贵州茅台",
                        "period": "2024",
                        "metric": "营收",
                        "value": "1740",
                        "turn": 1,
                    }
                ],
                new_open_items=["不该出现的上一轮未完成"],
            )
        ]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=llm,
        turn=2,
        run_id="r",
        token_limit=20,
        allow_llm=True,
        policy=CompactPolicy(context_window=40, retain_ratio=0.2, prune_chars=10_000),
    )
    events = await store.load_events(header.session_id)
    structured = next(event for event in events if event.event_type == "compaction/summary").data[
        "structured"
    ]
    assert structured["facts"][0]["company"] == "贵州茅台"
    assert structured["open_items"] == []


@pytest.mark.asyncio
async def test_free_text_fallback_then_narrative_merge():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await _seed_turn(store, header.session_id, 1, user="甲" * 40, tool="乙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "当前问题", "source": "user"},
        ),
    )
    first = FakeLlmAdapter(summaries=["较早已查乙公司相关材料。"])
    policy = CompactPolicy(context_window=40, retain_ratio=0.2, prune_chars=10_000)
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=first,
        turn=1,
        run_id="r1",
        token_limit=20,
        allow_llm=True,
        policy=policy,
    )
    first_summary = next(
        event
        for event in await store.load_events(header.session_id)
        if event.event_type == "compaction/summary"
    )
    assert first_summary.data["compaction_schema_version"] == 2
    assert "较早已查乙公司相关材料" in first_summary.data["structured"]["narrative"]
    await _seed_turn(store, header.session_id, 1, user="更多" * 20, tool="丙" * 40)
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "再问", "source": "user"},
        ),
    )
    second = FakeLlmAdapter(
        structured_deltas=[_delta(new_facts=[{"note": "丙公司材料", "turn": 1}])]
    )
    await maybe_compact(
        store=store,
        session_id=header.session_id,
        llm=second,
        turn=1,
        run_id="r2",
        token_limit=20,
        allow_llm=True,
        policy=policy,
    )
    events = await store.load_events(header.session_id)
    latest = latest_committed_summary(events)
    assert latest is not None
    structured = latest.data["structured"]
    assert "较早已查乙公司相关材料" in structured["narrative"]
    assert structured["facts"][0]["note"] == "丙公司材料"


def test_prune_facts_keeps_last_ten_turns_including_current():
    facts = [
        CompactionFact(company=f"c{index}", metric="营收", value=str(index), turn=index)
        for index in range(1, 13)
    ]
    kept = prune_facts(facts, turn=12, max_turns=10)
    turns = [item.turn for item in kept]
    assert 1 not in turns and 2 not in turns
    assert turns == list(range(3, 13))


def test_drop_oldest_turn_never_drops_current():
    facts = [
        CompactionFact(company="old", turn=10),
        CompactionFact(company="cur", turn=12),
    ]
    kept = drop_oldest_turn_facts(facts, turn=12)
    assert [item.company for item in kept] == ["cur"]
    again = drop_oldest_turn_facts(kept, turn=12)
    assert [item.company for item in again] == ["cur"]


def test_merge_summary_clips_narrative_and_same_turn_open_items():
    old = CompactionSummary(
        facts=[CompactionFact(company="甲", metric="营收", value="1", turn=1)],
        open_items=["净利润未查"],
        narrative="旧说明",
    )
    merged = merge_summary(
        old,
        _delta(
            new_facts=[{"company": "乙", "metric": "营收", "value": "2", "turn": 1}],
            new_completed_items=["净利润未查"],
            new_open_items=["估值未查"],
            narrative_delta="x" * 500,
        ),
        turn=1,
        same_turn=True,
        max_narrative_chars=400,
    )
    assert "净利润未查" not in merged.open_items
    assert "估值未查" in merged.open_items
    assert len(merged.narrative) == 400
    text = render_summary_text(merged)
    assert "已确认事实" in text
    assert "本轮未完成" in text


def test_request_tokens_includes_system_messages_and_tools():
    base = request_tokens(system="系统", messages=[], tools=[])
    with_message = request_tokens(
        system="系统", messages=[{"role": "user", "content": "问题" * 20}], tools=[]
    )
    with_tools = request_tokens(
        system="系统", messages=[], tools=[{"type": "function", "name": "search", "description": "检索" * 20}]
    )
    assert with_message > base
    assert with_tools > base


def test_v2_merge_resolves_items_and_completed_closes_open():
    old = CompactionSummaryV2(
        decisions=[CompactionItemV2(id="decision-1", content="使用旧来源")],
        open_items=[CompactionItemV2(id="open-1", content="核验利润")],
    )
    merged = merge_summary_v2(
        old,
        CompactionDeltaV2(
            resolved_ids=["decision-1"],
            new_completed_items=[CompactionItemV2(content="核验利润")],
        ),
        turn=2,
    )
    assert merged.decisions == []
    assert merged.open_items == []
    assert merged.completed_items[0].content == "核验利润"


def test_v2_shrink_preserves_protected_state():
    summary = CompactionSummaryV2(
        constraints=[
            CompactionItemV2(id="c", content="只能使用官方来源", priority="critical")
        ],
        open_items=[CompactionItemV2(id="o", content="核验利润", updated_turn=3)],
        conversation_notes=[
            CompactionItemV2(id=f"n-{index}", content="旧闲聊" * 30, updated_turn=1)
            for index in range(4)
        ],
    )
    shrunk = shrink_summary_v2(summary, turn=3, token_budget=20)
    assert [item.id for item in shrunk.constraints] == ["c"]
    assert [item.id for item in shrunk.open_items] == ["o"]
    assert shrunk.conversation_notes == []


@pytest.mark.asyncio
async def test_terminal_budget_error_has_no_retry_loop():
    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    await store.append(
        header.session_id,
        EventDraft(
            event_type="user/message",
            turn=1,
            surface_op="append",
            data={"content": "当前问题", "source": "user"},
        ),
    )

    class FixedCounter(FakeLlmAdapter):
        def count_request_tokens(self, **_: object) -> int:
            return 100

    with pytest.raises(ContextBudgetExhaustedError) as raised:
        await maybe_compact(
            store=store,
            session_id=header.session_id,
            llm=FixedCounter(),
            turn=1,
            run_id="r",
            token_limit=80,
            allow_llm=True,
            system="system",
            tools=[],
            enforce_budget=True,
            policy=CompactPolicy(context_window=100, trigger_ratio=0.8, target_ratio=0.6),
        )
    assert raised.value.code == "context_budget_exhausted"
    violations = [
        event for event in await store.load_events(header.session_id)
        if event.event_type == "invariant/violation"
    ]
    assert violations[-1].data["final_tokens"] == 100
