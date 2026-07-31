"""V2 多轮问题改写测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from agents.init_turn import begin_turn_workspace
from agents.orchestrator.analyzer.heuristic import latest_query
from agents.query_rewrite.node import query_rewrite_node


def test_turn_workspace_preserves_pending_query_clarification() -> None:
    assert "pending_query_clarification" not in begin_turn_workspace()


async def test_complete_question_with_history_does_not_call_rewrite_llm(
    monkeypatch,
) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="贵州茅台去年营收是多少？"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="宁德时代最近有哪些公告？"),
            ]
        }
    )

    assert result["rewritten_query"] == "宁德时代最近有哪些公告？"
    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_contextual_followup_uses_rewrite_llm(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="贵州茅台 2023 年营业收入是多少？"
    )
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="贵州茅台 2024 年营业收入是多少？"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="那 2023 年呢？"),
            ]
        }
    )

    assert result["rewritten_query"] == "贵州茅台 2023 年营业收入是多少？"
    assert result["rewrite_status"] == "rewrite"
    assert result["rewrite_resolution"]["resolved_entities"] == ["贵州茅台"]
    model.ainvoke.assert_awaited_once()


async def test_short_metric_followup_with_punctuation_uses_rewrite_llm(
    monkeypatch,
) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="贵州茅台市盈率是多少？")
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="分析贵州茅台"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="市盈率多少？"),
            ]
        }
    )

    assert result["rewrite_status"] == "rewrite"
    model.ainvoke.assert_awaited_once()


async def test_dangling_reference_without_history_is_uncertain(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="__UNCERTAIN__\n我需要先确认“它”具体指哪个对象。您想查哪只股票或基金？"
    )
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {"messages": [HumanMessage(content="那它去年营收多少？")]}
    )

    assert result["rewrite_status"] == "uncertain"
    assert "context_missing" in result["rewrite_reason_codes"]
    assert "具体指哪个对象" in result["rewrite_clarification_message"]
    model.ainvoke.assert_awaited_once()


async def test_unrelated_history_is_not_treated_as_usable_context(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="__UNCERTAIN__\n前面的天气对话没有提供金融对象。您想查哪只股票、基金或指数？"
    )
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="北京今天天气怎么样？"),
                AIMessage(content="天气回答"),
                HumanMessage(content="那它去年营收多少？"),
            ]
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "context_irrelevant" in result["rewrite_reason_codes"]
    assert result["pending_query_clarification"]["original_query"] == "那它去年营收多少？"
    model.ainvoke.assert_awaited_once()


async def test_complete_question_with_discourse_prefix_passthroughs(
    monkeypatch,
) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="上一轮问题"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="另外，请解释市盈率是什么？"),
            ]
        }
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_complete_entity_metric_question_passthroughs(monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="上一轮问题"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="贵州茅台市盈率多少？"),
            ]
        }
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_first_turn_complete_general_question_passthroughs(monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {"messages": [HumanMessage(content="北京天气怎么样？")]}
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_open_financial_research_question_passthroughs(monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {"messages": [HumanMessage(content="白酒龙头去年表现怎么样？")]}
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_temporal_demonstrative_is_not_treated_as_entity_reference(
    monkeypatch,
) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="上一轮问题"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="这个季度贵州茅台营收多少？"),
            ]
        }
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_social_ack_with_followup_prefix_does_not_rewrite(monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="上一轮问题"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="再见"),
            ]
        }
    )

    assert result["rewrite_status"] == "passthrough"
    model.ainvoke.assert_not_awaited()


async def test_singular_reference_with_multiple_entities_is_uncertain(
    monkeypatch,
) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="__UNCERTAIN__\n上文有贵州茅台和五粮液。您说的是哪一家公司？"
    )
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)
    summary = {
        "schema_version": "2.0",
        "revision": 1,
        "active_topic_id": "comparison",
        "topics": [
            {
                "topic_id": "comparison",
                "title": "白酒公司比较",
                "domain": "finance",
                "status": "active",
                "last_touched_revision": 1,
                "finance_context": {
                    "securities": ["贵州茅台", "五粮液"],
                    "time_ranges": [],
                    "metrics": ["净利润"],
                },
            }
        ],
    }

    result = await query_rewrite_node(
        {
            "conversation_summary_v2": summary,
            "messages": [HumanMessage(content="它去年利润多少？")],
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "multiple_entity_candidates" in result["rewrite_reason_codes"]
    model.ainvoke.assert_awaited_once()


async def test_recent_dialogue_multiple_entities_is_uncertain_without_summary(
    monkeypatch,
) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content="__UNCERTAIN__\n上文有贵州茅台和五粮液。您说的是哪一家公司？"
    )
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="比较贵州茅台和五粮液"),
                AIMessage(content="上一轮比较结果"),
                HumanMessage(content="它去年利润多少？"),
            ]
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "multiple_entity_candidates" in result["rewrite_reason_codes"]
    model.ainvoke.assert_awaited_once()


async def test_rewrite_model_failure_is_uncertain(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="分析贵州茅台"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="那2023年呢？"),
            ]
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "model_error" in result["rewrite_reason_codes"]
    assert result["rewrite_failure_kind"] == "provider"
    model.ainvoke.assert_awaited_once()


async def test_rewrite_model_can_reject_ambiguous_recent_context(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="__UNCERTAIN__")
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="比较贵州茅台和五粮液"),
                AIMessage(content="上一轮比较结果"),
                HumanMessage(content="其中哪家风险更大？"),
            ]
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "model_uncertain" in result["rewrite_reason_codes"]
    model.ainvoke.assert_awaited_once()


async def test_rewrite_output_must_resolve_to_context_candidate(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="五粮液 2023 年营业收入是多少？")
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "messages": [
                HumanMessage(content="分析贵州茅台"),
                AIMessage(content="上一轮回答"),
                HumanMessage(content="那 2023 年营收呢？"),
            ]
        }
    )

    assert result["rewrite_status"] == "uncertain"
    assert "rewrite_validation_failed" in result["rewrite_reason_codes"]
    assert result["rewrite_resolution"]["resolved_entities"] == []


async def test_clarification_reply_creates_new_complete_query(monkeypatch) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="贵州茅台去年营业收入是多少？")
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "pending_query_clarification": {
                "kind": "query_rewrite",
                "original_query": "它去年营收多少？",
                "missing_fields": ["query_target"],
                "candidate_entities": [],
                "remaining_turns": 2,
            },
            "messages": [
                HumanMessage(content="它去年营收多少？"),
                AIMessage(content="请明确公司名称。"),
                HumanMessage(content="贵州茅台"),
            ],
        }
    )

    assert result["rewrite_status"] == "rewrite"
    assert result["rewritten_query"] == "贵州茅台去年营业收入是多少？"
    assert result["pending_query_clarification"] == {}
    assert result["rewrite_resolution"]["resolved_entities"] == ["贵州茅台"]


async def test_complete_new_question_discards_pending_clarification(monkeypatch) -> None:
    model = AsyncMock()
    monkeypatch.setattr("agents.query_rewrite.node.get_router_llm", lambda: model)

    result = await query_rewrite_node(
        {
            "pending_query_clarification": {
                "original_query": "它去年营收多少？",
                "remaining_turns": 2,
            },
            "messages": [HumanMessage(content="北京天气怎么样？")],
        }
    )

    assert result["rewrite_status"] == "passthrough"
    assert result["pending_query_clarification"] == {}
    model.ainvoke.assert_not_awaited()


def test_analyzer_prefers_rewritten_query() -> None:
    assert (
        latest_query(
            {
                "messages": [HumanMessage(content="那 2023 年呢？")],
                "rewritten_query": "贵州茅台 2023 年营业收入是多少？",
                "rewrite_status": "rewrite",
            }
        )
        == "贵州茅台 2023 年营业收入是多少？"
    )
