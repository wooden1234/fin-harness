"""Orchestrator Analyzer 语义契约与确定性 Resolver 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agents.orchestrator.analyzer.heuristic import heuristic_profile
from agents.orchestrator.analyzer.node import (
    _analyzer_error_details,
    analyze_request,
    build_analyzer_envelope,
)
from agents.orchestrator.analyzer.prompts import (
    ANALYZER_REPAIR_SYSTEM_PROMPT,
    build_analyzer_system_prompt,
)
from agents.orchestrator.analyzer.schema import AnalyzerOutput
from agents.orchestrator.analyzer.validate import (
    assert_plan_capabilities,
    assert_task_capabilities,
    validate_and_normalize,
)
from agents.orchestrator.capability_resolver import resolve_finance_capabilities
from agents.orchestrator.contracts import ExecutionDecision, RequestProfile, TaskSpec
from agents.orchestrator.planner import build_plan_from_profile
from agents.structured_output import ainvoke_json_output


def test_open_financial_question_uses_deep_research() -> None:
    profile = heuristic_profile("白酒龙头去年表现怎么样？")
    plan = build_plan_from_profile(profile)

    assert profile.intents == ["open_research"]
    assert profile.entities == ["白酒龙头"]
    assert profile.constraints["entity_scope_type"] == "dynamic_group"
    assert profile.execution.mode == "deep_research"
    assert profile.execution.budget_tier == "research"
    assert "local_documents" in profile.execution.data_sources
    assert plan.tasks[0].agent_id == "research_workflow"
    assert_plan_capabilities(plan)


def test_compound_stock_screening_is_resolved_to_deep_research() -> None:
    profile = heuristic_profile("筛选新能源股票并分析前三只的风险")
    plan = build_plan_from_profile(profile)

    assert profile.intents == ["stock_screening", "open_research"]
    assert profile.execution.mode == "deep_research"
    assert [task.agent_id for task in plan.tasks] == ["research_workflow"]


@pytest.mark.parametrize(
    "query",
    ["贵州茅台最近为什么涨？", "新能源行业未来怎么样？"],
)
def test_open_attribution_and_industry_outlook_do_not_require_clarification(
    query: str,
) -> None:
    profile = heuristic_profile(query)

    assert profile.intents == ["open_research"]
    assert profile.execution.mode == "deep_research"
    assert profile.missing_fields == []


def test_analyzer_prompt_only_asks_for_semantics() -> None:
    prompt = build_analyzer_system_prompt()

    assert "白酒龙头去年表现怎么样" in prompt
    assert "clarification_message" in prompt
    assert "不选择 Agent、Tool、数据源、执行模式或预算" in prompt
    assert "合法 JSON 对象" in prompt
    assert '"normalized_query"' in prompt
    assert "行业或板块必须使用 dynamic_group" in prompt
    assert "合法 JSON 对象" in ANALYZER_REPAIR_SYSTEM_PROMPT
    assert "行业、板块和主题使用 dynamic_group" in ANALYZER_REPAIR_SYSTEM_PROMPT
    assert "rationale" in ANALYZER_REPAIR_SYSTEM_PROMPT


def test_analyzer_error_details_extracts_provider_fields_and_redacts_secrets() -> None:
    class ProviderBadRequest(Exception):
        status_code = 400
        body = {
            "error": {
                "type": "invalid_request_error",
                "code": "json_mode_invalid",
                "param": "messages",
                "message": "Prompt must mention JSON; token sk-secret12345678",
            }
        }

    details = _analyzer_error_details(ProviderBadRequest())

    assert details["status_code"] == "400"
    assert details["provider_type"] == "invalid_request_error"
    assert details["provider_code"] == "json_mode_invalid"
    assert details["param"] == "messages"
    assert details["message"] == "Prompt must mention JSON; token [REDACTED]"


def test_analyzer_error_details_safely_diagnoses_parser_output() -> None:
    class OutputParserException(Exception):
        observation = "Invalid JSON returned by provider"
        llm_output = '{"intents": ["not_allowed"]}'

    details = _analyzer_error_details(OutputParserException("invalid json"))

    assert details["parser_reason"] == "invalid_json_output"
    assert details["output_length"] == "28"
    assert len(details["output_sha256"]) == 16
    assert details["message"] == "invalid_json_output"
    assert "not_allowed" not in details["message"]


def test_analyzer_error_details_prioritizes_schema_validation() -> None:
    class OutputParserException(Exception):
        llm_output = '{"constraints": {"entity_scope_type": "industry"}}'

    error = OutputParserException(
        "Failed to parse JSON. Got: 1 validation error for AnalyzerOutput"
    )
    details = _analyzer_error_details(error)

    assert details["parser_reason"] == "schema_validation_failed"
    assert details["message"] == "schema_validation_failed"


def test_analyzer_output_normalizes_industry_scope_to_dynamic_group() -> None:
    result = AnalyzerOutput.model_validate(
        {
            "normalized_query": "存储芯片价格接近峰值，现在还能介入吗？",
            "intents": ["open_research"],
            "freshness_required": True,
            "entities": ["存储芯片"],
            "constraints": {
                "entity_scope_type": "industry",
                "analysis_dimensions": ["价格位置", "投资介入时机"],
            },
        }
    )

    assert result.constraints.entity_scope_type == "dynamic_group"


def test_structured_output_recovers_valid_json_from_parser_exception() -> None:
    class ParserError(Exception):
        llm_output = '{"intents": ["general_chat"]}'

    class FakeRunnable:
        async def ainvoke(self, messages):
            del messages
            raise ParserError("parser wrapper failed")

    class FakeModel:
        def with_structured_output(self, schema, method=None):
            assert schema is AnalyzerOutput
            assert method == "json_mode"
            return FakeRunnable()

    result = asyncio.run(
        ainvoke_json_output(
            FakeModel(),
            AnalyzerOutput,
            [("system", "输出 JSON"), ("human", "你好")],
        )
    )

    assert result.intents == ["general_chat"]


def test_analyzer_output_rejects_execution_fields() -> None:
    with pytest.raises(ValidationError, match="preferred_agent"):
        AnalyzerOutput.model_validate(
            {"intents": ["general_chat"], "preferred_agent": "general_agent"}
        )
    with pytest.raises(ValidationError, match="market_tool_id"):
        AnalyzerOutput.model_validate(
            {
                "intents": ["market_query"],
                "constraints": {"market_tool_id": "iwencai.market.query"},
            }
        )


def test_clarification_keeps_natural_model_message() -> None:
    raw = AnalyzerOutput(
        intents=["clarify"],
        missing_fields=["query_target"],
        clarification_message=(
            "我需要先确认具体对象。例如：**贵州茅台去年怎么样？** "
            "您想查哪只股票、基金或指数？"
        ),
    )
    result = validate_and_normalize(raw, original_query="它去年怎么样？")

    assert "贵州茅台去年怎么样" in result.profile.clarification_message
    assert result.profile.execution.mode == "clarify"


@pytest.mark.parametrize(
    ("query", "mode", "agent_id", "capability"),
    [
        ("贵州茅台营收多少", "structured_finance", "finance_agent", "financial_query"),
        ("什么是ROE", "faq_lookup", "finance_agent", "faq"),
        ("根据宁德时代2025年报查找风险因素原文", "document_qa", "finance_agent", "pdf"),
        ("查询沪深300今日涨跌幅", "market_acquire", "market_acquisition_workflow", "iwencai.index.query"),
        ("查询宁德时代最近的公告", "research_retrieve", "research_retrieval_workflow", "iwencai.announcement.search"),
        ("筛选成交量放大且涨跌幅为正的股票", "stock_screen", "stock_screening_agent", "iwencai.screen"),
    ],
)
def test_deterministic_execution_routes(
    query: str,
    mode: str,
    agent_id: str,
    capability: str,
) -> None:
    profile = heuristic_profile(query)
    plan = build_plan_from_profile(profile)

    assert profile.execution.mode == mode
    assert plan.tasks[0].agent_id == agent_id
    assert plan.tasks[0].required_capabilities == [capability]


def test_finance_capability_resolver_uses_only_resolver_authorization() -> None:
    profile = heuristic_profile("贵州茅台营收多少")

    assert resolve_finance_capabilities(profile) == ["financial_query"]
    assert resolve_finance_capabilities(profile, allowed_capabilities=["faq"]) == []


def test_finance_capability_resolver_fails_closed_without_authorization() -> None:
    profile = RequestProfile(
        original_query="回答这个问题",
        intents=["concept_explain"],
        execution=ExecutionDecision(mode="faq_lookup", budget_tier="standard"),
    )
    plan = build_plan_from_profile(profile)

    assert plan.tasks == []
    assert plan.metadata["planning_status"] == "uncovered"


def test_corporate_faq_scope_requires_explicit_enterprise_policy_semantics() -> None:
    fund = heuristic_profile("基金申购费怎么计算？")
    corporate = heuristic_profile("付款超过20万元应该谁审批？")

    assert fund.execution.knowledge_scope == ["capital_market"]
    assert corporate.execution.knowledge_scope == ["explicit:corporate_finance"]


@pytest.mark.parametrize(
    "query",
    ["帮我看看", "给点意见", "详细说说", "帮忙判断一下", "这个分析一下"],
)
def test_action_only_query_requires_clarification(query: str) -> None:
    profile = heuristic_profile(query)

    assert profile.intents == ["clarify"]
    assert profile.missing_fields == ["query_target"]
    assert profile.execution.mode == "clarify"


def test_candidate_compute_requires_an_artifact_or_plan() -> None:
    raw = AnalyzerOutput(
        normalized_query="从刚才候选股票中按营收增速排序取前5只",
        intents=["candidate_compute"],
    )
    result = validate_and_normalize(raw, original_query=raw.normalized_query)

    assert result.needs_repair is True
    assert "candidate_compute_requires_artifact_or_plan" in result.issues


def test_analyzer_envelope_projects_candidate_metadata_without_rows() -> None:
    envelope = build_analyzer_envelope(
        {
            "messages": [HumanMessage(content="刚才第二家公司呢？")],
            "agent_results": [
                {
                    "task_id": "screen",
                    "agent_id": "stock_screening_agent",
                    "status": "completed",
                    "structured_data": {
                        "dataset_id": "set-1",
                        "universe": "A股",
                        "as_of": "2026-07-31",
                        "rows": [{"ticker": "600519", "name": "贵州茅台"}],
                    },
                }
            ],
        }
    )

    assert envelope.artifacts[0].artifact_id == "set-1"
    assert envelope.artifacts[0].row_count == 1
    assert envelope.artifacts[0].available_fields == ["ticker", "name"]
    assert "贵州茅台" not in envelope.model_dump_json()


def test_capability_assert_rejects_unknown_capability() -> None:
    task = TaskSpec(
        task_id="bad",
        objective="x",
        agent_id="finance_agent",
        required_capabilities=["not_a_real_capability"],
    )
    with pytest.raises(ValueError, match="capability_mismatch"):
        assert_task_capabilities(task)


def test_validate_empty_profile_needs_repair() -> None:
    result = validate_and_normalize(AnalyzerOutput(), original_query="帮我看看")

    assert result.needs_repair is True
    assert "empty_intents" in result.issues


def test_analyze_request_uses_llm_semantics_and_local_resolution() -> None:
    raw = AnalyzerOutput(
        normalized_query="筛选低市盈率股票",
        intents=["stock_screening"],
        freshness_required=True,
    )
    with patch(
        "agents.orchestrator.analyzer.node.analyze_once",
        new=AsyncMock(return_value=raw),
    ):
        output = asyncio.run(
            analyze_request({"messages": [HumanMessage(content=raw.normalized_query)]})
        )

    assert output["request_profile"].execution.mode == "stock_screen"
    assert output["steps"] == ["orchestrator:analyze_request:llm"]


def test_analyze_request_falls_back_after_transient_errors() -> None:
    with patch(
        "agents.orchestrator.analyzer.node.analyze_once",
        new=AsyncMock(side_effect=TimeoutError("boom")),
    ):
        output = asyncio.run(
            analyze_request(
                {"messages": [HumanMessage(content="筛选新能源股票并分析风险")]}
            )
        )

    assert output["request_profile"].execution.mode == "deep_research"
    assert output["steps"] == ["orchestrator:analyze_request:heuristic_error_fallback"]


def test_analyze_request_repairs_once_then_falls_back() -> None:
    bad = AnalyzerOutput()
    repair = AsyncMock(return_value=bad)
    with (
        patch(
            "agents.orchestrator.analyzer.node.analyze_once",
            new=AsyncMock(return_value=bad),
        ),
        patch("agents.orchestrator.analyzer.node.repair_profile", new=repair),
    ):
        output = asyncio.run(
            analyze_request({"messages": [HumanMessage(content="帮我看看")]})
        )

    assert repair.await_count == 1
    assert output["request_profile"].execution.mode == "clarify"
    assert output["steps"] == ["orchestrator:analyze_request:heuristic_fallback"]
