"""V2 多轮问题改写测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from agents.orchestrator.analyzer.heuristic import latest_query
from agents.query_rewrite.node import query_rewrite_node


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
    assert result["rewrite_status"] == "success"
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

    assert result["rewrite_status"] == "success"
    model.ainvoke.assert_awaited_once()


def test_analyzer_prefers_successful_rewritten_query() -> None:
    assert (
        latest_query(
            {
                "messages": [HumanMessage(content="那 2023 年呢？")],
                "rewritten_query": "贵州茅台 2023 年营业收入是多少？",
                "rewrite_status": "success",
            }
        )
        == "贵州茅台 2023 年营业收入是多少？"
    )
