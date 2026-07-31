"""统一上下文空间预算和原子块裁剪测试。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.context_space import (
    ContextBudgetPolicy,
    ToolLoopContextGovernor,
    effective_context_limit,
    measure_context,
    snip_largest_removable_block,
)
from agents.context_space.governor import ToolLoopSummary
from agents.context_compressor.node import (
    _truncate_oversized_messages,
    select_keep_indices,
)
from agents.context_compressor.tokens import estimate_tokens, truncate_to_token_limit


class _ProfileModel:
    def __init__(self, max_input_tokens: int):
        self.profile = {"max_input_tokens": max_input_tokens}


class _StructuredModel(_ProfileModel):
    def __init__(self, output=None, error: Exception | None = None):
        super().__init__(10_000)
        self.output = output
        self.error = error
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages, config=None):
        del config
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def test_business_limit_does_not_grow_with_model_window() -> None:
    policy = ContextBudgetPolicy(explicit_max_tokens=16_000)

    assert effective_context_limit(policy, _ProfileModel(128_000)) == 16_000
    assert effective_context_limit(policy, _ProfileModel(8_000)) == 8_000


def test_approximate_measurement_applies_safety_multiplier() -> None:
    policy = ContextBudgetPolicy(
        explicit_max_tokens=100,
        trigger_ratio=0.75,
        target_ratio=0.50,
        admission_ratio=0.85,
        approximate_safety_multiplier=1.20,
    )
    measurement = measure_context(
        [HumanMessage(content="a" * 252)],
        policy=policy,
    )

    assert measurement.counter_kind == "approximate"
    assert measurement.estimated_tokens == 76
    assert measurement.trigger_exceeded is True
    assert measurement.admission_exceeded is False


def test_snip_preserves_current_question_and_system_message() -> None:
    messages = [
        SystemMessage(content="固定安全规则"),
        HumanMessage(content="很长的历史问题" * 100, id="old-human"),
        AIMessage(content="很长的历史回答" * 200, id="old-ai"),
        HumanMessage(content="当前问题", id="current-human"),
    ]

    updated, metadata = snip_largest_removable_block(messages)

    assert metadata.changed is True
    assert updated[0].content == "固定安全规则"
    assert updated[-1].content == "当前问题"
    assert "[历史内容已裁剪]" in updated[1].content
    assert "[历史内容已裁剪]" in updated[2].content


def test_snip_tool_block_preserves_protocol_and_recent_tool_round() -> None:
    old_call = AIMessage(
        content="",
        id="old-call",
        tool_calls=[{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}],
    )
    old_result = ToolMessage(
        content="x" * 8_000,
        id="old-result",
        name="lookup",
        tool_call_id="call-1",
    )
    recent_call = AIMessage(
        content="",
        id="recent-call",
        tool_calls=[{"name": "lookup", "args": {}, "id": "call-2", "type": "tool_call"}],
    )
    recent_result = ToolMessage(
        content="recent",
        id="recent-result",
        name="lookup",
        tool_call_id="call-2",
    )
    messages = [
        SystemMessage(content="系统"),
        HumanMessage(content="当前问题", id="human"),
        old_call,
        old_result,
        recent_call,
        recent_result,
    ]

    updated, metadata = snip_largest_removable_block(messages)

    assert metadata.block_type == "tool_call"
    assert isinstance(updated[3], ToolMessage)
    assert updated[3].tool_call_id == "call-1"
    assert updated[3].name == "lookup"
    assert "[工具结果已裁剪]" in updated[3].content
    assert updated[5].content == "recent"


async def test_tool_governor_summarizes_old_round_and_keeps_recent_round() -> None:
    model = _StructuredModel(
        ToolLoopSummary(
            objective="查询财务数据",
            completed_calls=["lookup"],
            key_results=["已取得历史数据"],
        )
    )
    policy = ContextBudgetPolicy(
        explicit_max_tokens=1_000,
        trigger_ratio=0.50,
        target_ratio=0.25,
        admission_ratio=0.90,
    )
    messages = [
        SystemMessage(content="系统"),
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[{"name": "lookup", "args": {}, "id": "old", "type": "tool_call"}],
        ),
        ToolMessage(content="x" * 3_000, tool_call_id="old"),
        AIMessage(
            content="",
            tool_calls=[{"name": "lookup", "args": {}, "id": "recent", "type": "tool_call"}],
        ),
        ToolMessage(content="recent", tool_call_id="recent"),
    ]
    governor = ToolLoopContextGovernor(llm=model, tools=[], policy=policy)

    updated, admitted = await governor.prepare(messages)

    assert admitted is True
    assert model.calls == 1
    assert governor.counters.compaction_round_count == 1
    assert any("已压缩的工具工作记录" in str(item.content) for item in updated)
    assert updated[-1].content == "recent"


async def test_tool_governor_repairs_then_snips_when_summary_fails() -> None:
    model = _StructuredModel(error=ValueError("bad schema"))
    policy = ContextBudgetPolicy(
        explicit_max_tokens=1_000,
        trigger_ratio=0.50,
        target_ratio=0.25,
        admission_ratio=0.90,
    )
    messages = [
        SystemMessage(content="系统"),
        HumanMessage(content="旧问题"),
        AIMessage(content="x" * 3_000),
        HumanMessage(content="当前问题"),
    ]
    governor = ToolLoopContextGovernor(llm=model, tools=[], policy=policy)

    updated, admitted = await governor.prepare(messages)

    assert admitted is True
    assert model.calls == 2
    assert governor.counters.summary_attempt_count == 2
    assert governor.counters.snip_count == 1
    assert "[历史内容已裁剪]" in updated[1].content


def test_conversation_retention_never_splits_a_human_ai_turn() -> None:
    messages = [
        SystemMessage(content="安全规则", id="system"),
        HumanMessage(content="旧问题一" * 100, id="human-1"),
        AIMessage(content="旧回答一" * 100, id="ai-1"),
        HumanMessage(content="旧问题二" * 100, id="human-2"),
        AIMessage(content="旧回答二" * 100, id="ai-2"),
        HumanMessage(content="当前问题", id="current"),
    ]

    keep = set(select_keep_indices(messages, message_token_budget=260))

    assert 0 in keep
    assert 5 in keep
    assert (1 in keep) == (2 in keep)
    assert (3 in keep) == (4 in keep)


def test_oversized_tool_message_update_preserves_protocol_fields() -> None:
    message = ToolMessage(
        content="x" * 20_000,
        id="tool-result",
        name="lookup",
        tool_call_id="call-1",
    )

    updates = _truncate_oversized_messages([message])

    assert len(updates) == 1
    assert isinstance(updates[0], ToolMessage)
    assert updates[0].id == "tool-result"
    assert updates[0].name == "lookup"
    assert updates[0].tool_call_id == "call-1"


def test_truncation_suffix_is_included_in_token_limit() -> None:
    clipped = truncate_to_token_limit("中" * 100, 10)

    assert clipped.endswith("…")
    assert estimate_tokens(clipped) <= 10


def test_provider_retry_limit_is_per_model_invocation() -> None:
    model = _StructuredModel()
    policy = ContextBudgetPolicy(
        explicit_max_tokens=1_000,
        trigger_ratio=0.50,
        target_ratio=0.25,
        admission_ratio=0.90,
        max_provider_retries_per_invocation=1,
    )
    governor = ToolLoopContextGovernor(llm=model, tools=[], policy=policy)
    first = [
        SystemMessage(content="系统"),
        HumanMessage(content="旧问题"),
        AIMessage(content="x" * 2_000),
        HumanMessage(content="当前问题一"),
    ]
    second = [
        SystemMessage(content="系统"),
        HumanMessage(content="另一旧问题"),
        AIMessage(content="y" * 2_000),
        HumanMessage(content="当前问题二"),
    ]

    _, first_retry = governor.provider_overflow_recovery(first)
    _, second_retry = governor.provider_overflow_recovery(second)

    assert first_retry is True
    assert second_retry is True
    assert governor.counters.provider_retry_count == 2
