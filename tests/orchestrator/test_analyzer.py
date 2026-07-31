"""Orchestrator Analyzer 单元与 mock 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.orchestrator.analyzer.heuristic import heuristic_profile
from agents.orchestrator.analyzer.node import analyze_request
from agents.orchestrator.analyzer.schema import AnalyzerOutput
from agents.orchestrator.analyzer.validate import (
    assert_plan_capabilities,
    assert_task_capabilities,
    validate_and_normalize,
)
from agents.orchestrator.contracts import RequestProfile, TaskSpec
from agents.orchestrator.capability_resolver import resolve_finance_capabilities
from agents.orchestrator.planner import build_plan_from_profile


def test_heuristic_compound_stock_query_uses_deep_research():
    profile = heuristic_profile("筛选新能源股票并分析前三只的风险")
    plan = build_plan_from_profile(profile)
    assert profile.operation_type == "deep_research"
    assert profile.preferred_agent == "research_workflow"
    assert [task.agent_id for task in plan.tasks] == ["research_workflow"]
    assert plan.tasks[0].depends_on == []
    assert_plan_capabilities(plan)


def test_heuristic_finance_query():
    profile = heuristic_profile("贵州茅台营收多少")
    plan = build_plan_from_profile(profile)
    assert [task.task_id for task in plan.tasks] == ["finance"]
    assert_plan_capabilities(plan)


def test_finance_capability_resolver_selects_structured_metric():
    profile = heuristic_profile("贵州茅台营收多少")
    plan = build_plan_from_profile(profile)

    assert plan.tasks[0].required_capabilities == ["financial_query"]


def test_finance_capability_resolver_selects_pdf_for_annual_report():
    profile = RequestProfile(
        original_query="根据贵州茅台年报分析现金流",
        normalized_query="根据贵州茅台年报分析现金流",
        intents=["financial_research"],
        data_sources=["finance_rag"],
        preferred_agent="finance_agent",
    )
    plan = build_plan_from_profile(profile)

    assert plan.tasks[0].required_capabilities == ["pdf"]


def test_finance_capability_resolver_fails_closed_for_no_source():
    profile = RequestProfile(
        original_query="回答这个问题",
        normalized_query="回答这个问题",
        intents=["financial_research"],
        data_sources=["none"],
        preferred_agent="finance_agent",
    )
    plan = build_plan_from_profile(profile)

    assert plan.tasks == []
    assert plan.metadata == {
        "planning_status": "uncovered",
        "error_code": "no_finance_capability_in_scope",
    }


def test_finance_capability_resolver_intersects_allowed_scope():
    profile = heuristic_profile("贵州茅台营收多少")

    assert resolve_finance_capabilities(profile, allowed_capabilities=["faq"]) == []


def test_heuristic_metric_question_is_finance_not_stock():
    profile = heuristic_profile("什么是ROE")
    assert profile.preferred_agent == "finance_agent"
    assert "stock_screening" not in profile.intents


def test_heuristic_market_query_selects_dedicated_tool():
    profile = heuristic_profile("查询沪深300今日涨跌幅")
    plan = build_plan_from_profile(profile)

    assert profile.preferred_agent == "market_acquisition_workflow"
    assert profile.data_sources == ["market"]
    assert profile.operation_type == "acquire"
    assert profile.constraints["market_tool_id"] == "iwencai.index.query"
    assert plan.tasks[0].required_capabilities == ["iwencai.index.query"]
    assert plan.tasks[0].input_data == {"market_tool_id": "iwencai.index.query"}


def test_stock_screening_takes_priority_over_embedded_market_fields():
    profile = heuristic_profile("筛选成交量放大且涨跌幅为正的股票")
    plan = build_plan_from_profile(profile)

    assert profile.preferred_agent == "stock_screening_agent"
    assert plan.tasks[0].agent_id == "stock_screening_agent"


def test_heuristic_ambiguous_query_goes_general():
    profile = heuristic_profile("帮我看看")
    assert profile.preferred_agent == "general_agent"
    assert profile.intents == ["general_chat"]


def test_heuristic_industry_without_finance_marker_goes_general():
    profile = heuristic_profile("新能源未来怎么样")
    assert profile.preferred_agent == "general_agent"


def test_capability_assert_rejects_unknown_capability():
    task = TaskSpec(
        task_id="bad",
        objective="x",
        agent_id="finance_agent",
        required_capabilities=["not_a_real_capability"],
    )
    with pytest.raises(ValueError, match="capability_mismatch"):
        assert_task_capabilities(task)


def test_validate_normalizes_stock_intent_to_stock_screening_agent():
    raw = AnalyzerOutput(
        normalized_query="帮我选新能源股票",
        intents=["stock_screening"],
        complexity="single_capability",
        preferred_agent="finance_agent",
    )
    result = validate_and_normalize(raw, original_query="帮我选新能源股票")
    assert result.profile.preferred_agent == "stock_screening_agent"
    assert result.needs_repair is False
    assert "stock_screening_requires_stock_screening_agent" in result.issues


def test_validate_keeps_metric_question_on_finance_agent():
    raw = AnalyzerOutput(
        normalized_query="贵州茅台的市盈率是多少",
        intents=["financial_research"],
        complexity="single_capability",
        preferred_agent="finance_agent",
    )
    result = validate_and_normalize(raw, original_query="贵州茅台的市盈率是多少")
    assert result.profile.preferred_agent == "finance_agent"
    assert result.needs_repair is False


def test_validate_empty_profile_needs_repair():
    raw = AnalyzerOutput(complexity="simple")
    result = validate_and_normalize(raw, original_query="帮我看看")
    assert result.needs_repair is True
    assert "empty_intents" in result.issues


def test_build_plan_supports_general_agent():
    profile = RequestProfile(
        original_query="你好",
        normalized_query="你好",
        intents=["general_chat"],
        complexity="simple",
        preferred_agent="general_agent",
    )
    plan = build_plan_from_profile(profile)
    assert plan.tasks[0].agent_id == "general_agent"
    assert_plan_capabilities(plan)


def test_analyze_request_uses_llm_when_valid():
    raw = AnalyzerOutput(
        normalized_query="筛选低市盈率股票",
        intents=["stock_screening"],
        complexity="single_capability",
        preferred_agent="stock_screening_agent",
        freshness_required=True,
    )
    with patch(
        "agents.orchestrator.analyzer.node.analyze_once",
        new=AsyncMock(return_value=raw),
    ):
        output = asyncio.run(
            analyze_request({"messages": [HumanMessage(content="筛选低市盈率股票")]})
        )
    assert output["request_profile"].preferred_agent == "stock_screening_agent"
    assert output["steps"] == ["orchestrator:analyze_request:llm"]


def test_analyze_request_falls_back_on_llm_error():
    with patch(
        "agents.orchestrator.analyzer.node.analyze_once",
        new=AsyncMock(side_effect=TimeoutError("boom")),
    ):
        output = asyncio.run(
            analyze_request(
                {"messages": [HumanMessage(content="筛选新能源股票并分析风险")]}
            )
        )
    assert output["request_profile"].preferred_agent == "research_workflow"
    assert output["steps"] == [
        "orchestrator:analyze_request:heuristic_error_fallback"
    ]


def test_analyze_request_empty_query_skips_llm():
    with patch(
        "agents.orchestrator.analyzer.node.analyze_once",
        new=AsyncMock(),
    ) as mocked:
        output = asyncio.run(analyze_request({"messages": []}))
    mocked.assert_not_called()
    assert output["request_profile"].missing_fields == ["query"]
    assert output["steps"] == ["orchestrator:analyze_request:empty"]


def test_analyze_request_repairs_then_falls_back_when_still_invalid():
    bad = AnalyzerOutput(complexity="simple", preferred_agent=None, intents=[])
    with (
        patch(
            "agents.orchestrator.analyzer.node.analyze_once",
            new=AsyncMock(return_value=bad),
        ),
        patch(
            "agents.orchestrator.analyzer.node.repair_profile",
            new=AsyncMock(return_value=bad),
        ),
    ):
        output = asyncio.run(
            analyze_request({"messages": [HumanMessage(content="帮我看看")]})
        )
    assert output["steps"] == ["orchestrator:analyze_request:heuristic_fallback"]
    assert output["request_profile"].preferred_agent == "general_agent"


def test_analyze_request_max_two_llm_calls_skips_repair_after_transient_retry():
    bad = AnalyzerOutput(complexity="simple", preferred_agent=None, intents=[])
    analyze_mock = AsyncMock(side_effect=[TimeoutError("boom"), bad])
    repair_mock = AsyncMock()
    with (
        patch(
            "agents.orchestrator.analyzer.node.analyze_once",
            new=analyze_mock,
        ),
        patch(
            "agents.orchestrator.analyzer.node.repair_profile",
            new=repair_mock,
        ),
    ):
        output = asyncio.run(
            analyze_request({"messages": [HumanMessage(content="帮我看看")]})
        )
    assert analyze_mock.await_count == 2
    repair_mock.assert_not_called()
    assert output["steps"] == ["orchestrator:analyze_request:heuristic_fallback"]
    assert output["request_profile"].preferred_agent == "general_agent"


def test_heuristic_research_query_uses_research_retrieval_workflow():
    profile = heuristic_profile("查询宁德时代最近的公告")
    plan = build_plan_from_profile(profile)

    assert profile.preferred_agent == "research_retrieval_workflow"
    assert profile.data_sources == ["research"]
    assert profile.operation_type == "retrieve"
    assert plan.tasks[0].agent_id == "research_retrieval_workflow"
    assert plan.tasks[0].input_data == {
        "research_tool_id": "iwencai.announcement.search"
    }


def test_heuristic_bare_report_keyword_uses_report_search():
    """裸词“研报”曾经无法命中 iwencai.report.search，误判为 finance_agent。"""
    for query in ("查询研报", "新能源研报", "帮我看下贵州茅台的研报", "宁德时代最新研报"):
        profile = heuristic_profile(query)
        plan = build_plan_from_profile(profile)

        assert profile.preferred_agent == "research_retrieval_workflow", query
        assert plan.tasks[0].input_data == {
            "research_tool_id": "iwencai.report.search"
        }, query


def test_heuristic_rating_keyword_takes_priority_over_bare_report():
    """“研报评级”应命中机构评级 Tool，不能被裸词“研报”抢先匹配。"""
    profile = heuristic_profile("看一下机构对贵州茅台的研报评级")
    plan = build_plan_from_profile(profile)

    assert profile.preferred_agent == "research_retrieval_workflow"
    assert plan.tasks[0].input_data == {"research_tool_id": "iwencai.rating.query"}


def test_heuristic_upstream_filter_uses_market_compute():
    profile = heuristic_profile("从刚才候选股票中按营收增速排序取前5只")
    plan = build_plan_from_profile(profile)

    assert profile.preferred_agent == "market.compute"
    assert profile.data_sources == ["upstream_data"]
    assert profile.operation_type == "compute"
    assert plan.tasks[0].agent_id == "market.compute"
