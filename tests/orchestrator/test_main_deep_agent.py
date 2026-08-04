"""Main DeepAgent 动态预算与证据门测试。"""

import json
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.main_deep_agent.assembly import MainDeepAgentAssembly, run_main_deep_agent
from agents.main_deep_agent.node import main_deep_agent_node
from agents.main_deep_agent.prompts import build_main_system_prompt
from tools.web_domains import DEFAULT_WEB_SEARCH_DOMAINS, resolve_search_domains
from agents.main_deep_agent.middleware.budget import MainAgentBudgetController
from agents.main_deep_agent.middleware.finalization import (
    MainAgentFailureFinalizationMiddleware,
)
from agents.main_deep_agent.middleware.quality import MainEvidenceQualityMiddleware
from agents.main_deep_agent.contracts import (
    MainAgentResponse,
    MainAgentStatement,
    MainAgentTable,
    MainAgentTableRow,
)
from agents.main_deep_agent.tools.evidence_adapter import (
    detect_fact_conflicts,
    web_evidence_from_payload,
)
from agents.main_deep_agent.quality import (
    evaluate_main_response,
    main_evidence_quality_gate,
)
from agents.main_deep_agent.state import MainAgentProgressJournal
from agents.main_deep_agent.tools.catalog import (
    MAIN_TOOL_ARGS_SCHEMAS,
    MAIN_TOOL_IDS,
)
from agents.main_deep_agent.tools.factory import build_main_tools, resolve_main_tool_ids
from agents.orchestrator.contracts import Evidence
from agents.runtime_context import AgentRuntimeContext
from app.api.agent_progress import (
    build_step_event,
    build_todo_snapshot_event,
    extract_agent_todos_snapshot,
    label_for_public_step,
    sanitize_step_detail,
)
from agents.main_deep_agent.tools.step_detail import build_public_tool_step_detail
from agents.main_deep_agent.quality.enrichments import (
    build_answer_charts,
    sanitize_follow_ups,
)
from tools.calculation import run_calculation
from tools.finance import fetch_financial_fact, lookup_financial_fact
from tools.knowledge import lookup_knowledge_fact
from tools.core.base import ToolResult


def _evidence(evidence_id: str, family: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        task_id="main",
        source_type=family,
        provider=family,
        title=family,
        content=f"{family} 已核验事实",
        observed_at="2026-08-01T00:00:00+00:00",
        confidence=0.8,
        metadata={
            "source_family": family,
            "displayable": True,
            "display_text": f"{family} 已核验事实",
        },
    )


def test_deep_agent_assembly_maps_declared_components(monkeypatch) -> None:
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-main-agent"

    monkeypatch.setattr("deepagents.create_deep_agent", fake_create_deep_agent)
    model = SimpleNamespace()
    middleware = SimpleNamespace()
    assembly = MainDeepAgentAssembly(
        model=model,
        tools=(),
        system_prompt="测试系统提示词",
        middleware=(middleware,),
        skills=("/skills/main",),
    )

    assert assembly.create() == "compiled-main-agent"
    assert captured["model"] is model
    assert captured["tools"] == []
    assert captured["system_prompt"] == "测试系统提示词"
    assert captured["middleware"] == [middleware]
    assert captured["skills"] == ["/skills/main"]
    assert captured["subagents"] is None
    assert captured["memory"] is None
    assert captured["name"] == "main_deep_agent"


def test_compiled_deep_agent_has_only_main_quality_after_agent_hook() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="测试")]))
    quality = MainEvidenceQualityMiddleware(
        journal=MainAgentProgressJournal(),
        budget=MainAgentBudgetController(started_monotonic=time.monotonic()),
        investment_action_sensitive=False,
    )
    agent = MainDeepAgentAssembly(
        model=model,
        tools=(),
        system_prompt="测试系统提示词",
        middleware=(quality,),
        response_format=ToolStrategy(MainAgentResponse, handle_errors=False),
    ).create()

    after_agent_nodes = [
        node for node in agent.get_graph().nodes if node.endswith(".after_agent")
    ]
    assert after_agent_nodes == ["MainEvidenceQualityMiddleware.after_agent"]


def test_main_prompt_maps_dynamic_memory_preferences_to_output_contract() -> None:
    prompt = build_main_system_prompt(investment_action_sensitive=False)

    assert "以「小财」自称" in prompt
    assert "preferred_output_format=table" in prompt
    assert "不得强行比较" not in prompt
    assert "最近完整财年" in prompt
    assert "按公司年结分别取值" in prompt
    assert "禁止为对齐把 3 月年结公司压回更早一年" in prompt
    assert "截至2026-03-31" in prompt or "2026-03-31" in prompt
    assert "response_detail_level=brief" in prompt
    assert "根级 tables" in prompt
    assert "table row 引用真实 Evidence ID" in prompt
    assert "简单单点事实无需 todos" in prompt
    assert "3–5 条可执行 todos" in prompt
    assert "禁止空泛" in prompt
    assert "异年结必须分开查" in prompt
    assert "苹果约 9 月末" in prompt
    assert "不用网页金额覆盖财务主表" in prompt
    assert "传闻/政策/审批提速" in prompt
    assert "调用 `search_web` 前必须把用户口语改写成检索问句" in prompt
    assert "禁止把多槽位糊成一句超长搜索" in prompt
    assert "板块领涨" in prompt
    assert "只用 `query_iwencai_industry`" in prompt
    assert "最近已结束交易日" in prompt
    assert "session_phase=" in prompt
    assert "单一结构化事实不为凑来源" in prompt
    assert "不使用口号式总结" in prompt
    assert "statement_type=caveat" in prompt
    assert "不要写「主表：」" in prompt
    assert "不为对齐财年或币种继续 Web" in prompt
    assert "query_iwencai_finance" in prompt
    assert "同口径年结多实体必须合查" in prompt
    assert "含「累计」" in prompt or "累计" in prompt
    assert "禁止用单季度或最新价" in prompt
    assert "catalog_pdf_knowledge_tool` 每任务最多 1 次" in prompt
    assert "禁止同族反复改参重试" in prompt
    assert "有累计财务 Evidence 后应立即 MainAgentResponse 成稿" in prompt
    assert "screen_iwencai_usstock" in prompt
    assert "search_iwencai_announcement` 只检索公告文档" in prompt
    assert "禁止塞入营业收入/归母净利润" in prompt
    assert "total=0" in prompt
    assert "主表保留财务 Skill 原值" in prompt
    assert "禁止用中期或其他未结束期间填主表" in prompt
    assert "不强行换算到同一财年或同一币种" in prompt
    assert "follow_ups" in prompt


def test_cn_equity_session_context_before_close_uses_previous_weekday() -> None:
    from zoneinfo import ZoneInfo

    from agents.main_deep_agent.prompts import resolve_cn_equity_session_context

    # 2026-08-05 00:08 上海：盘前，最近已结束交易日应为 2026-08-04
    now = datetime(2026, 8, 5, 0, 8, tzinfo=ZoneInfo("Asia/Shanghai"))
    ctx = resolve_cn_equity_session_context(now=now)
    assert ctx["session_phase"] == "before_close"
    assert ctx["last_session_date"] == "2026-08-04"
    assert ctx["last_session_ymd"] == "20260804"
    assert "2026年8月4日" in ctx["last_session_cn"]


@pytest.mark.asyncio
async def test_brief_grounded_renders_prose_without_heading_or_bullets() -> None:
    evidence = _evidence("e-brief", "financial")
    evidence.metadata["entity"] = "川金诺"
    response = MainAgentResponse(
        mode="grounded",
        heading="川金诺增长情况",
        statements=[
            MainAgentStatement(
                text=(
                    "川金诺（300505）2025年前三季度归母净利润同比增长"
                    "175.61%，归母净利润为3.04亿元。"
                ),
                evidence_ids=[evidence.evidence_id],
            )
        ],
        follow_ups=[
            "川金诺同期营业收入是多少？",
            "现在还能买入吗？",
        ],
    )
    update = await main_evidence_quality_gate(
        {
            "messages": [
                HumanMessage(content="川金诺2025年前三季度净利润同比增长多少？")
            ],
            "main_agent_response": response,
            "evidence": [evidence],
        }
    )
    summary = update["summary"]
    assert "### " not in summary
    assert "- 川金诺" not in summary
    assert summary.startswith("**川金诺")
    assert "175.61%" in summary
    assert "[1]" not in summary
    assert update["answer_follow_ups"] == ["川金诺同期营业收入是多少？"]
    # 单期事实不出图
    assert update["answer_charts"] == []


@pytest.mark.asyncio
async def test_main_node_injects_current_memory_preferences(monkeypatch) -> None:
    captured_messages: list[list[object]] = []

    async def fake_run_main_deep_agent(*, messages, context, **_kwargs):
        captured_messages.append(list(messages))
        return (
        MainAgentResponse(mode="direct", direct_answer="测试回答"),
        MainAgentProgressJournal(),
        MainAgentBudgetController(started_monotonic=context.started_monotonic),
        "completed",
        "",
        )

    monkeypatch.setattr(
        "agents.main_deep_agent.node.run_main_deep_agent",
        fake_run_main_deep_agent,
    )
    base_state = {"messages": [HumanMessage(content="比较两家公司")]}
    await main_deep_agent_node(
        {**base_state, "memory_context": {"preferred_output_format": "table"}}
    )
    await main_deep_agent_node(
        {**base_state, "memory_context": {"preferred_output_format": "markdown"}}
    )
    await main_deep_agent_node({**base_state, "memory_context": {}})

    table_context = "\n".join(
        str(message.content)
        for message in captured_messages[0]
        if isinstance(message, SystemMessage)
    )
    markdown_context = "\n".join(
        str(message.content)
        for message in captured_messages[1]
        if isinstance(message, SystemMessage)
    )
    cleared_context = "\n".join(
        str(message.content)
        for message in captured_messages[2]
        if isinstance(message, SystemMessage)
    )
    assert "preferred_output_format=table" in table_context
    assert "preferred_output_format=markdown" in markdown_context
    assert "preferred_output_format=table" not in markdown_context
    assert "preferred_output_format" not in cleared_context


def test_main_table_requires_consistent_column_count() -> None:
    with pytest.raises(ValueError, match="table_row_column_count_mismatch"):
        MainAgentTable(
        columns=["公司", "营收"],
        rows=[MainAgentTableRow(cells=["腾讯"], evidence_ids=["e1"])],
        )


def test_journal_normalizes_agent_todos_for_observability() -> None:
    journal = MainAgentProgressJournal()
    journal.set_agent_todos(
        [
        {"content": "  查询财报  ", "status": "in_progress", "extra": "hidden"},
        {"content": "x" * 400, "status": "completed"},
        {"content": "非法状态", "status": "failed"},
        {"status": "pending"},
        ]
    )

    assert journal.snapshot()["agent_todos"] == [
        {"content": "查询财报", "status": "in_progress"},
        {"content": "x" * 300, "status": "completed"},
    ]


@pytest.mark.asyncio
async def test_run_main_deep_agent_captures_final_todos(monkeypatch) -> None:
    class FakeAgent:
        async def ainvoke(self, _input, *, config):
            assert config["recursion_limit"] > 0
            return {
                "messages": [],
                "todos": [
                    {"content": "查询天气", "status": "completed"},
                    {"content": "整理答案", "status": "in_progress"},
                ],
                "structured_response": MainAgentResponse(
                    mode="direct",
                    direct_answer="测试回答",
                ),
            }

    monkeypatch.setattr(
        "agents.main_deep_agent.assembly.build_main_deep_agent",
        lambda **_kwargs: FakeAgent(),
    )
    response, journal, _budget, status, error = await run_main_deep_agent(
        messages=[HumanMessage(content="测试")],
        context=AgentRuntimeContext(),
        investment_action_sensitive=False,
    )

    assert response is not None
    assert status == "completed"
    assert error == ""
    assert journal.agent_todos == [
        {"content": "查询天气", "status": "completed"},
        {"content": "整理答案", "status": "in_progress"},
    ]


@pytest.mark.asyncio
async def test_run_main_deep_agent_discards_stale_quality_response(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self, journal: MainAgentProgressJournal) -> None:
            self.journal = journal

        async def ainvoke(self, _input, *, config):
            assert config["recursion_limit"] > 0
            self.journal.quality_revision_count = 1
            self.journal.quality_revision_outcome = "no_new_structured_response"
            return {
                "messages": [AIMessage(content="只返回了普通文本")],
                "structured_response": MainAgentResponse(
                    mode="direct",
                    direct_answer="这是残留的旧结果",
                ),
            }

    monkeypatch.setattr(
        "agents.main_deep_agent.assembly.build_main_deep_agent",
        lambda **kwargs: FakeAgent(kwargs["journal"]),
    )
    response, journal, _budget, status, error = await run_main_deep_agent(
        messages=[HumanMessage(content="测试")],
        context=AgentRuntimeContext(),
        investment_action_sensitive=False,
    )

    assert response is None
    assert status == "structured_output_failed"
    assert error == "quality_revision_missing_structured_response"
    assert journal.quality_revision_outcome == "no_new_structured_response"


def test_todo_snapshot_event_uses_full_replacement_payload() -> None:
    raw_update = {
        "tools": {
        "todos": [
            {"content": "查询天气", "status": "completed"},
            {"content": "整理答案", "status": "in_progress"},
        ]
        }
    }
    snapshot = extract_agent_todos_snapshot(raw_update)
    event = build_todo_snapshot_event(snapshot)

    assert snapshot == [
        {"content": "查询天气", "status": "completed"},
        {"content": "整理答案", "status": "in_progress"},
    ]
    assert event["type"] == "todo_snapshot"
    assert [item["status"] for item in event["todos"]] == [
        "completed",
        "in_progress",
    ]
    assert all(item["id"].startswith("main-todo-") for item in event["todos"])


def test_public_step_labels_use_narrative_copy() -> None:
    assert label_for_public_step("problem_analysis", "running") == "正在理解问题…"
    assert label_for_public_step("problem_analysis", "done") == "已理解问题"
    assert label_for_public_step("web_search", "done") == "已取得联网资料"
    assert "已完成" not in label_for_public_step("evidence_validation", "done")


def test_main_tool_catalog_keeps_specialized_argument_contracts() -> None:
    assert "web.search" in MAIN_TOOL_IDS
    assert "iwencai.query" in MAIN_TOOL_IDS
    assert "iwencai.finance.query" in MAIN_TOOL_IDS
    assert "iwencai.usstock.screen" in MAIN_TOOL_IDS
    assert "knowledge.fact.lookup" in MAIN_TOOL_IDS
    assert "finance.fact.lookup" not in MAIN_TOOL_IDS
    assert "finance.query_advanced" not in MAIN_TOOL_IDS
    assert set(MAIN_TOOL_ARGS_SCHEMAS) == {
        "web.search",
        "knowledge.faq.search",
        "knowledge.pdf.search",
        "calculation.run",
    }
    web_args = MAIN_TOOL_ARGS_SCHEMAS["web.search"]
    with pytest.raises(ValueError):
        web_args.model_validate({"query": "苹果2025财年年度报告"})
    validated = web_args.model_validate(
        {"query": "苹果2025财年年度报告", "entities": ["苹果公司"]}
    )
    assert validated.entities == ["苹果公司"]


def test_runtime_permissions_reduce_visible_tool_catalog() -> None:
    context = AgentRuntimeContext(
        permissions=("web.search", "knowledge.fact.lookup"),
    )
    assert resolve_main_tool_ids(context) == (
        "web.search",
        "knowledge.fact.lookup",
    )
    assert resolve_main_tool_ids(
        AgentRuntimeContext(permissions=("*",))
    ) == MAIN_TOOL_IDS


def test_budget_upgrades_from_actual_source_families() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    budget.register("web.search")
    assert budget.budget_tier == "standard"
    budget.register("calculation.run")
    assert budget.budget_tier == "standard"
    budget.register("iwencai.market.query")
    assert budget.budget_tier == "deep_research"
    assert budget.independent_source_families == {"web", "market"}


def test_quality_revision_reserve_tracks_dynamic_budget_window() -> None:
    class TimeoutScope:
        def __init__(self, deadline: float) -> None:
            self.deadline = deadline

        def when(self) -> float:
            return self.deadline

    budget = MainAgentBudgetController(
        started_monotonic=time.monotonic(),
        soft_seconds=12.0,
        hard_seconds=21.0,
    )
    scope = TimeoutScope(time.monotonic() + 9.1)
    budget.timeout_scope = scope  # type: ignore[assignment]

    assert budget.quality_revision_reserve_seconds() == 9.0
    assert budget.can_start_quality_revision() is True
    scope.deadline = time.monotonic() + 8.9
    assert budget.can_start_quality_revision() is False


def test_quality_evaluation_counts_tables_but_not_caveats() -> None:
    evidence = _evidence("e-table", "financial")
    response = MainAgentResponse(
        mode="grounded",
        heading="结果",
        statements=[
            MainAgentStatement(
                text="请结合自身风险承受能力。",
                statement_type="caveat",
            )
        ],
        tables=[
            MainAgentTable(
                columns=["公司", "结果"],
                rows=[
                    MainAgentTableRow(
                        cells=["示例公司", "已核验"],
                        evidence_ids=["e-table"],
                    )
                ],
            )
        ],
    )

    evaluation = evaluate_main_response(response, [evidence], sensitive=False)

    assert evaluation.accepted_evidence_content_count == 1
    assert evaluation.quality_report.passed is True


@pytest.mark.asyncio
async def test_quality_middleware_revises_once_with_existing_evidence() -> None:
    evidence = _evidence("e-valid", "financial")
    journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    budget = MainAgentBudgetController(
        started_monotonic=time.monotonic(),
        soft_seconds=10.0,
        hard_seconds=20.0,
    )
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=budget,
        investment_action_sensitive=False,
    )
    rejected = MainAgentResponse(
        mode="grounded",
        
        heading="结论",
        statements=[
            MainAgentStatement(
                text="引用不存在的证据。",
                evidence_ids=["missing"],
            )
        ],
    )

    update = await middleware.aafter_agent(
        {"messages": [HumanMessage(content="测试")], "structured_response": rejected},
        None,
    )

    assert update is not None
    assert update["jump_to"] == "model"
    assert budget.stop_new_tools is True
    assert budget.finalization_reason == "quality_revision_rewrite_only"
    assert journal.quality_revision_count == 1
    feedback = update["messages"][0]
    assert feedback.additional_kwargs["lc_source"] == "main_quality_guard"

    accepted = MainAgentResponse(
        mode="grounded",
        
        heading="结论",
        statements=[
            MainAgentStatement(
                text="这是已有证据支持的结果。",
                evidence_ids=["e-valid"],
            )
        ],
    )
    call_id = "main-response-call"
    second_messages = [
        HumanMessage(content="测试"),
        feedback,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "MainAgentResponse",
                    "args": accepted.model_dump(),
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Returning structured response",
            tool_call_id=call_id,
            name="MainAgentResponse",
        ),
    ]
    second_update = await middleware.aafter_agent(
        {"messages": second_messages, "structured_response": accepted},
        None,
    )

    assert second_update is None
    assert journal.quality_revision_outcome == "revised_passed"
    assert journal.quality_after_supported_count == 1
    assert await middleware.aafter_agent(
        {"messages": second_messages, "structured_response": accepted},
        None,
    ) is None


@pytest.mark.asyncio
async def test_quality_middleware_rejects_stale_structured_response() -> None:
    evidence = _evidence("e-valid", "financial")
    journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=MainAgentBudgetController(
        started_monotonic=time.monotonic(),
        soft_seconds=10.0,
        hard_seconds=20.0,
        ),
        investment_action_sensitive=False,
    )
    rejected = MainAgentResponse(
        mode="grounded",
                heading="结论",
        statements=[MainAgentStatement(text="无效引用", evidence_ids=["missing"])],

    )
    update = await middleware.aafter_agent(
        {"messages": [HumanMessage(content="测试")], "structured_response": rejected},
        None,
    )
    assert update is not None

    second_update = await middleware.aafter_agent(
        {
        "messages": [
            HumanMessage(content="测试"),
            update["messages"][0],
            AIMessage(content="没有重新生成结构化输出"),
        ],
        "structured_response": rejected,
        },
        None,
    )

    assert second_update is None
    assert journal.quality_revision_outcome == "no_new_structured_response"


@pytest.mark.asyncio
async def test_quality_middleware_stops_after_fresh_invalid_revision() -> None:
    evidence = _evidence("e-valid", "financial")
    journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=MainAgentBudgetController(
            started_monotonic=time.monotonic(),
            soft_seconds=10.0,
            hard_seconds=20.0,
        ),
        investment_action_sensitive=False,
    )
    rejected = MainAgentResponse(
        mode="grounded",
        
        heading="结论",
        statements=[
            MainAgentStatement(text="无效引用", evidence_ids=["missing"])
        ],
    )
    update = await middleware.aafter_agent(
        {"messages": [HumanMessage(content="测试")], "structured_response": rejected},
        None,
    )
    assert update is not None

    feedback = update["messages"][0]
    call_id = "fresh-invalid-response"
    second_update = await middleware.aafter_agent(
        {
            "messages": [
                HumanMessage(content="测试"),
                feedback,
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "MainAgentResponse",
                            "args": rejected.model_dump(),
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Returning structured response",
                    tool_call_id=call_id,
                    name="MainAgentResponse",
                ),
            ],
            "structured_response": rejected,
        },
        None,
    )

    assert second_update is None
    assert journal.quality_revision_outcome == "revision_failed"
    assert journal.quality_revision_count == 1


@pytest.mark.asyncio
async def test_quality_middleware_does_not_revise_partial_valid_answer() -> None:
    evidence = _evidence("e-valid", "financial")
    journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    budget = MainAgentBudgetController(
        started_monotonic=time.monotonic(),
        soft_seconds=10.0,
        hard_seconds=20.0,
    )
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=budget,
        investment_action_sensitive=False,
    )
    response = MainAgentResponse(
        mode="grounded",
                    heading="结论",
            statements=[
                MainAgentStatement(text="有效事实", evidence_ids=["e-valid"]),
                MainAgentStatement(text="无效事实", evidence_ids=["missing"]),
            ],
    )

    update = await middleware.aafter_agent(
        {"messages": [HumanMessage(content="测试")], "structured_response": response},
        None,
    )

    assert update is None
    assert journal.quality_before_supported_count == 1
    assert journal.quality_revision_count == 0
    assert journal.quality_revision_outcome == "not_needed"
    assert budget.stop_new_tools is False


@pytest.mark.asyncio
async def test_quality_middleware_sensitive_direct_requires_evidence_and_time() -> None:
    evidence = _evidence("e-valid", "financial")
    journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    response = MainAgentResponse(mode="direct", direct_answer="建议立即买入。")
    enough_budget = MainAgentBudgetController(
        started_monotonic=time.monotonic(),
        soft_seconds=10.0,
        hard_seconds=20.0,
    )
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=enough_budget,
        investment_action_sensitive=True,
    )

    update = await middleware.aafter_agent(
        {"messages": [HumanMessage(content="现在能买吗？")], "structured_response": response},
        None,
    )
    assert update is not None
    assert update["jump_to"] == "model"

    expired_journal = MainAgentProgressJournal(evidence={evidence.evidence_id: evidence})
    expired_budget = MainAgentBudgetController(
        started_monotonic=time.monotonic() - 15.0,
        soft_seconds=10.0,
        hard_seconds=20.0,
    )
    expired_middleware = MainEvidenceQualityMiddleware(
        journal=expired_journal,
        budget=expired_budget,
        investment_action_sensitive=True,
    )
    expired_update = await expired_middleware.aafter_agent(
        {"messages": [HumanMessage(content="现在能买吗？")], "structured_response": response},
        None,
    )
    assert expired_update is None
    assert expired_journal.quality_revision_outcome == "skipped_no_time"


@pytest.mark.asyncio
async def test_sensitive_direct_without_evidence_skips_revision() -> None:
    journal = MainAgentProgressJournal()
    middleware = MainEvidenceQualityMiddleware(
        journal=journal,
        budget=MainAgentBudgetController(started_monotonic=time.monotonic()),
        investment_action_sensitive=True,
    )

    update = await middleware.aafter_agent(
        {
        "messages": [HumanMessage(content="现在能买吗？")],
        "structured_response": MainAgentResponse(
            mode="direct",
            direct_answer="建议立即买入。",
        ),
        },
        None,
    )

    assert update is None
    assert journal.quality_revision_outcome == "skipped_no_evidence"


@pytest.mark.asyncio
async def test_research_with_one_family_salvages_without_action_conclusion() -> None:
    evidence = _evidence("e-web", "web")
    response = MainAgentResponse(
        mode="grounded",
        gaps=["当前资料不足以判断是否买入。"],
        
        heading="结论",
        statements=[
            MainAgentStatement(
                text="建议立即买入并加仓。",
                statement_type="inference",
                evidence_ids=["e-web"],
            )
        ],
    )
    update = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="存储芯片接近峰值，现在还能介入吗？")],
            "main_agent_response": response,
            "main_agent_journal": {"source_families": ["web"]},
            "investment_action_sensitive": True,
            "evidence": [evidence],
        }
    )
    assert update["execution_mode"] == "partial"
    assert "买入" not in update["summary"]
    assert "加仓" not in update["summary"]
    assert "第二个独立来源" in update["summary"]


@pytest.mark.asyncio
async def test_grounded_statements_require_real_evidence_ids() -> None:
    response = MainAgentResponse(
        mode="grounded",
                heading="事实",
        statements=[
            MainAgentStatement(text="无法核验的事实", evidence_ids=["missing"])
        ],

    )
    update = await main_evidence_quality_gate(
        {
        "messages": [HumanMessage(content="查询事实")],
        "main_agent_response": response,
        "main_agent_journal": {"source_families": ["knowledge"]},
        "evidence": [_evidence("e-faq", "knowledge")],
        }
    )
    assert update["execution_status"] == "completed_with_gaps"
    assert "无法核验的事实" not in update["summary"]
    assert "knowledge 已核验事实" in update["summary"]


@pytest.mark.asyncio
async def test_grounded_table_renders_only_rows_with_real_evidence() -> None:
    response = MainAgentResponse(
        mode="grounded",
        
        heading="营业收入对比",
        tables=[
            MainAgentTable(
                title="最近完整财年",
                columns=["公司", "营业收入"],
                rows=[
                    MainAgentTableRow(
                        cells=["腾讯", "6600 亿元"],
                        evidence_ids=["e-tencent"],
                    ),
                    MainAgentTableRow(
                        cells=["阿里", "未知数据"],
                        evidence_ids=["missing"],
                    ),
                ],
            )
        ],
    )
    update = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="比较营业收入")],
            "main_agent_response": response,
            "evidence": [_evidence("e-tencent", "financial")],
        }
    )

    assert "| 公司 | 营业收入 |" in update["summary"]
    assert "| 腾讯 | 6600 亿元 |" in update["summary"]
    assert "[1]" not in update["summary"]
    assert "未知数据" not in update["summary"]
    assert update["execution_status"] == "completed_with_gaps"
    assert update["citations"][0]["evidence_id"] == "e-tencent"
    assert "table_row_missing_evidence" in update["quality_report"].missing_evidence


def test_table_row_allows_empty_evidence_ids_for_parse_then_quality_drop() -> None:
    """模型偶发交出空 evidence_ids 时不应在结构化解析阶段整单失败。"""
    row = MainAgentTableRow.model_validate(
        {"cells": ["美团", "未知"], "evidence_ids": []}
    )
    assert row.evidence_ids == []

    response = MainAgentResponse.model_validate(
        {
            "mode": "grounded",
            "heading": "结论",
            "tables": [
                {
                    "title": "对比",
                    "columns": ["公司", "营收"],
                    "rows": [
                        {
                            "cells": ["腾讯", "1"],
                            "evidence_ids": ["e-tencent"],
                        },
                        {"cells": ["美团", "未知"], "evidence_ids": []},
                    ],
                }
            ],
        }
    )
    assert response.tables[0].rows[1].evidence_ids == []


def test_drop_unknown_top_level_keys_falls_back_to_salvage_not_silent_fold() -> None:
    """未知顶层字段（如 sections）直接丢弃，不做语义折回；已知字段仍可解析。"""
    response = MainAgentResponse.model_validate(
        {
            "mode": "grounded",
            "heading": "结论",
            "statements": [
                {
                    "text": "微软营收为正。",
                    "statement_type": "fact",
                    "evidence_ids": ["main:msft"],
                }
            ],
            "tables": [
                {
                    "title": "最近完整财年财务对比",
                    "columns": ["公司", "营收"],
                    "rows": [
                        {
                            "cells": ["微软", "3318亿"],
                            "evidence_ids": ["main:msft"],
                        }
                    ],
                }
            ],
            "gaps": ["苹果数值缺失"],
            "follow_ups": ["需要补充苹果数据吗？"],
            # 模型又发明了 sections：应被丢弃，不折回
            "sections": [
                {
                    "heading": "不应出现",
                    "statements": [
                        {
                            "text": "这段应被丢弃。",
                            "evidence_ids": ["main:drop"],
                        }
                    ],
                }
            ],
            "invented_field": {"foo": 1},
        }
    )
    assert response.heading == "结论"
    assert len(response.statements) == 1
    assert response.statements[0].text == "微软营收为正。"
    assert len(response.tables) == 1
    assert response.tables[0].title == "最近完整财年财务对比"
    assert response.gaps == ["苹果数值缺失"]
    assert response.follow_ups == ["需要补充苹果数据吗？"]
    dumped = response.model_dump()
    assert "sections" not in dumped
    assert "invented_field" not in dumped
    assert all(s.text != "这段应被丢弃。" for s in response.statements)


@pytest.mark.asyncio
async def test_comparison_section_renders_conclusion_table_then_caveat() -> None:
    """比较题：结论在前、表格居中、caveat 口径差异在表后；去掉重复表题。"""
    evidence = _evidence("e-moutai", "financial")
    evidence.metadata["entity"] = "贵州茅台"
    response = MainAgentResponse(
        mode="grounded",
        heading="主要财务数据对比（2025年度）",
        statements=[
            MainAgentStatement(
                text="两家公司2025年度均实现盈利，茅台营收与利润规模更大。",
                statement_type="fact",
                evidence_ids=["e-moutai"],
            ),
            MainAgentStatement(
                text="茅台区分营业总收入与营业收入；五粮液两者相同。",
                statement_type="caveat",
                evidence_ids=["e-moutai"],
            ),
        ],
        tables=[
            MainAgentTable(
                title="主表：最近一个完整财年财务对比",
                columns=["公司", "营业收入", "归母净利润"],
                rows=[
                    MainAgentTableRow(
                        cells=["贵州茅台", "1688亿元", "823亿元"],
                        evidence_ids=["e-moutai"],
                    ),
                ],
            )
        ],
    )
    update = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="比较茅台五粮液")],
            "main_agent_response": response,
            "evidence": [evidence],
        }
    )
    summary = update["summary"]
    conclusion_at = summary.index("均实现盈利")
    table_at = summary.index("最近一个完整财年财务对比")
    caveat_at = summary.index("区分营业总收入")
    assert conclusion_at < table_at < caveat_at
    assert "主表：" not in summary
    assert "### 主要财务数据对比" not in summary


@pytest.mark.asyncio
async def test_sensitive_question_cannot_use_direct_action_answer() -> None:
    update = await main_evidence_quality_gate(
        {
            "messages": [HumanMessage(content="现在能买吗？")],
            "main_agent_response": MainAgentResponse(
                mode="direct",
                direct_answer="建议立即买入。",
            ),
            "main_agent_journal": {"source_families": []},
            "investment_action_sensitive": True,
            "evidence": [],
        }
    )
    assert update["execution_mode"] == "clarify"
    assert "买入" not in update["summary"]


@pytest.mark.asyncio
async def test_calculation_executes_resolved_batch() -> None:
    rejected = await run_calculation.ainvoke(
        {"calculations": []}
    )
    assert rejected["ok"] is False
    accepted = await run_calculation.ainvoke(
        {
            "calculations": [{
                "calculation_id": "growth",
                "operation": "change_rate",
                "current_value": 120,
                "reference_value": 100,
                "input_evidence_ids": ["e-current", "e-reference"],
                "operand_refs": [
                    {"evidence_id": "e-current", "fact_index": 0},
                    {"evidence_id": "e-reference", "fact_index": 0},
                ],
            }],
        }
    )
    assert accepted["calculations"][0]["value"] == 20.0
    assert accepted["calculations"][0]["formula"]


@pytest.mark.asyncio
async def test_main_calculation_resolves_evidence_facts_and_blocks_duplicate() -> None:
    current = _evidence("e-current", "financial")
    current.metadata["facts"] = [{
        "entity": "甲公司",
        "metric": "revenue",
        "fiscal_period": "FY2025 Q4",
        "value": 120.0,
        "unit": "CNY",
        "currency": "CNY",
    }]
    reference = _evidence("e-reference", "financial")
    reference.metadata["facts"] = [{
        "entity": "甲公司",
        "metric": "revenue",
        "fiscal_period": "FY2024 Q4",
        "value": 100.0,
        "unit": "CNY",
        "currency": "CNY",
    }]
    journal = MainAgentProgressJournal()
    journal.record(
        tool_id="knowledge.fact.lookup",
        source_family="financial",
        status="completed",
        duration_ms=1.0,
        evidence=[current, reference],
    )
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    tools = build_main_tools(
        context=AgentRuntimeContext(permissions=("calculation.run",)),
        budget=budget,
        journal=journal,
    )
    calculation = {item.name: item for item in tools}["run_calculation"]
    arguments = {
        "calculations": [{
            "calculation_id": "growth",
            "operation": "change_rate",
            "current": {"evidence_id": "e-current", "fact_index": 0},
            "reference": {"evidence_id": "e-reference", "fact_index": 0},
        }]
    }

    result = await calculation.ainvoke(arguments)
    duplicate = await calculation.ainvoke(arguments)

    assert result["ok"] is True, result
    assert result["evidence"][0]["metadata"]["input_evidence_ids"] == [
        "e-current",
        "e-reference",
    ]
    assert budget.tool_counts["calculation"] == 1
    assert duplicate["error"] == "duplicate_tool_call"
    assert duplicate["retryable"] is False


def test_calculation_statement_requires_existing_operand_facts() -> None:
    current = _evidence("e-current", "financial")
    current.metadata["facts"] = [{"value": 120.0}]
    reference = _evidence("e-reference", "financial")
    reference.metadata["facts"] = [{"value": 100.0}]
    calculation = _evidence("e-calculation", "calculation")
    calculation.source_type = "calculation.run"
    calculation.metadata.update(
        {
            "formula": "(current_value-reference_value)/abs(reference_value)*100",
            "input_evidence_ids": ["e-current", "e-reference"],
            "operand_refs": [
                {"evidence_id": "e-current", "fact_index": 0},
                {"evidence_id": "e-reference", "fact_index": 0},
            ],
        }
    )
    response = MainAgentResponse(
        mode="grounded",
        statements=[MainAgentStatement(
            text="增速为20%。",
            statement_type="calculation",
            evidence_ids=["e-calculation"],)],
    )

    accepted = evaluate_main_response(
        response,
        [current, reference, calculation],
        sensitive=False,
    )
    rejected = evaluate_main_response(
        response,
        [current, calculation],
        sensitive=False,
    )

    assert accepted.quality_report.passed is True
    assert rejected.quality_report.passed is False
    assert "invalid_calculation_evidence" in rejected.quality_report.missing_evidence


@pytest.mark.asyncio
async def test_narrow_finance_serializes_only_fact_fields(monkeypatch) -> None:
    fact = SimpleNamespace(
        table=None,
        title="示例公司 Annual Report",
        ticker="000001",
        period_year=2025,
        fiscal_year=2025,
        metric=SimpleNamespace(canonical_name="净资产收益率"),
        metric_name="净资产收益率",
        raw_value="12.5",
        value=None,
        unit="%",
        currency="",
        source="annual-report.pdf",
        page_num=10,
    )

    async def fake_execute_query(question, intent):
        return [fact], "latest_lookup"

    monkeypatch.setattr(
        "tools.finance.FinancialFactService.execute_query",
        fake_execute_query,
    )
    result = await lookup_knowledge_fact.ainvoke(
        {
            "question": "示例公司最新 ROE",
            "companies": ["示例公司"],
            "metrics": ["净资产收益率"],
            "operation": "latest",
        }
    )
    assert result["ok"] is True
    assert result["facts"] == [
        {
            "company": "示例公司",
            "period_year": 2025,
            "metric": "净资产收益率",
            "value": "12.5",
            "unit": "%",
            "currency": "",
        }
    ]
    # 兼容入口仍可用。
    legacy = await lookup_financial_fact.ainvoke(
        {
            "question": "示例公司最新 ROE",
            "companies": ["示例公司"],
            "metrics": ["净资产收益率"],
            "operation": "latest",
        }
    )
    assert legacy["ok"] is True


@pytest.mark.asyncio
async def test_knowledge_fact_lookup_miss_returns_local_store_error(monkeypatch) -> None:
    async def fake_execute_query(question, intent):
        return [], "generic_search_safe"

    monkeypatch.setattr(
        "tools.finance.FinancialFactService.execute_query",
        fake_execute_query,
    )
    result = await fetch_financial_fact(
        "未知公司 ROE",
        companies=["未知公司"],
        metrics=["净资产收益率"],
        operation="latest",
    )
    assert result["ok"] is False
    assert result["error"] == "fact_not_in_local_store"


@pytest.mark.asyncio
async def test_knowledge_fact_lookup_rejects_non_whitelisted_query_shape(monkeypatch) -> None:
    async def fake_execute_query(question, intent):
        return [], "text_to_sql_fallback"

    monkeypatch.setattr(
        "tools.finance.FinancialFactService.execute_query",
        fake_execute_query,
    )
    monkeypatch.setattr(
        "tools.finance.FinancialFactService.TEXT_TO_SQL_FALLBACK_ROUTE",
        "text_to_sql_fallback",
    )

    result = await fetch_financial_fact(
        "比较甲乙公司两年多项指标",
        companies=["甲公司", "乙公司"],
        metrics=["营业收入", "净利润"],
        years=[2024, 2025],
        operation="compare",
    )

    assert result["ok"] is False
    assert result["error"] == "fact_query_scope_unsupported"
    assert result["route"] == "unsupported_structured_query"
    assert "不会生成 SQL" in result["message"]


@pytest.mark.asyncio
async def test_failed_tool_middleware_returns_finalization_message() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    budget.stop_new_tools = True
    business_tool = SimpleNamespace(name="search_web")
    response_tool = SimpleNamespace(name="MainAgentResponse")
    middleware = MainAgentFailureFinalizationMiddleware(
        budget,
        business_tool_names={"search_web"},
    )

    class Request:
        tools = [business_tool, response_tool]
        system_message = "原始指令"

        def override(self, **kwargs):
            return SimpleNamespace(**kwargs)

    async def handler(request):
        return request

    response = await middleware.awrap_model_call(Request(), handler)
    assert response.tools == [response_tool]
    assert "MainAgentResponse" in response.system_message


@pytest.mark.asyncio
async def test_finalization_forces_when_idle_with_cumulative_finance_facts() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    budget.tool_calls = 1
    journal = MainAgentProgressJournal()
    evidence = _evidence("e-cum", "iwencai.finance.query")
    evidence.metadata["facts"] = [
        {
            "entity": "苹果",
            "metric": "营业收入(累计)[20251231]",
            "fiscal_period": "FY2025 Q4",
            "value": 416161000000.0,
            "unit": "",
        },
        {
            "entity": "微软",
            "metric": "归属于母公司股东的净利润(累计)[20251231]",
            "fiscal_period": "FY2025 Q4",
            "value": 101832000000.0,
            "unit": "",
        },
    ]
    journal.evidence[evidence.evidence_id] = evidence
    middleware = MainAgentFailureFinalizationMiddleware(
        budget,
        journal=journal,
        business_tool_names={"query_iwencai_finance"},
        idle_wraps_with_evidence=2,
    )

    class Request:
        tools = [SimpleNamespace(name="query_iwencai_finance"), SimpleNamespace(name="MainAgentResponse")]
        system_message = "原始指令"

        def override(self, **kwargs):
            return SimpleNamespace(**kwargs)

    async def handler(request):
        return request

    first = await middleware.awrap_model_call(Request(), handler)
    assert budget.stop_new_tools is False
    assert len(first.tools) == 2

    second = await middleware.awrap_model_call(Request(), handler)
    assert budget.stop_new_tools is True
    assert budget.finalization_reason == "idle_with_finance_evidence"
    assert [tool.name for tool in second.tools] == ["MainAgentResponse"]
    assert "累计" in second.system_message


def test_iwencai_normalize_prefers_cumulative_over_quote_and_quarter() -> None:
    from tools.iwencai import _normalize_iwencai_payload

    payload = _normalize_iwencai_payload(
        {
            "ok": True,
            "data": {
                "datas": [
                    {
                        "股票简称": "苹果",
                        "最新价": 303.42,
                        "最新涨跌幅": -1.777,
                        "营业收入(累计)[20251231]": 416161000000.0,
                        "归属于母公司股东的净利润(累计)[20251231]": 112010000000.0,
                        "营业收入(单季度)[20251231]": 102466000000.0,
                    }
                ]
            },
        }
    )
    metrics = [item["metric"] for item in payload["data"]["facts"]]
    assert "营业收入(累计)[20251231]" in metrics
    assert "归属于母公司股东的净利润(累计)[20251231]" in metrics
    assert "最新价" not in metrics
    assert "最新涨跌幅" not in metrics
    assert metrics.index("营业收入(累计)[20251231]") < metrics.index(
        "营业收入(单季度)[20251231]"
    )


def test_web_results_are_split_and_aggregate_answer_is_not_evidence() -> None:
    evidence = web_evidence_from_payload(
        {
            "answer": "聚合答案不得成为证据",
            "configured": True,
            "results": [
                {
                    "title": "Apple reports fourth quarter results",
                    "url": "https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/",
                    "content": "Apple fiscal 2025 fourth quarter Services revenue grew 15%.",
                    "published_date": "2025-10-30",
                },
                {
                    "title": "分析一",
                    "url": "https://example.com/one",
                    "content": "Apple Services revenue grew 14% in FY2025 Q4.",
                    "published_date": "2025-10-31",
                },
                {
                    "title": "分析二",
                    "url": "https://example.org/two",
                    "content": "Apple FY2025 Q4 iPhone revenue growth was 6%.",
                    "published_date": "2025-11-01",
                },
            ],
        },
        entity="Apple",
    )
    assert len(evidence) == 3
    assert all("聚合答案" not in item.content for item in evidence)
    assert evidence[0].metadata["publisher_domain"] == "apple.com"
    assert evidence[0].metadata["source_grade"] == "official"
    assert evidence[0].metadata["fiscal_period"] == "FY2025 Q4"
    assert evidence[0].published_at == "2025-10-30"


def test_web_budget_does_not_depend_on_inferred_entities() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    assert budget.web_limit == 6
    for _ in range(2):
        assert budget.authorize("web.search", entity="NVIDIA")[0] is True
        budget.register("web.search", entity="NVIDIA")
    assert budget.authorize("web.search", entity="AMD")[0] is True


def test_web_family_budget_exhausted_only_stops_web_family() -> None:
    entities = ("甲公司", "乙公司", "丙公司")
    budget = MainAgentBudgetController(
        started_monotonic=time.monotonic(),
    )
    assert budget.web_limit == 6
    for entity in entities:
        assert budget.authorize("web.search", entity=entity)[0] is True
        budget.register("web.search", entity=entity)
    for entity in entities:
        assert budget.authorize("web.search", entity=entity)[0] is True
        budget.register("web.search", entity=entity)
    allowed, reason = budget.authorize("web.search", entity="甲公司")
    assert allowed is False
    assert reason == "tool_family_budget_exhausted:web"
    assert budget.stop_new_tools is False
    assert budget.finalization_reason == ""
    assert budget.authorize("iwencai.query", entity="甲公司")[0] is True


def test_repeated_family_budget_rejects_force_finalization() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    for _ in range(2):
        assert budget.authorize("knowledge.pdf.catalog")[0] is True
        budget.register("knowledge.pdf.catalog")

    first_ok, first_reason = budget.authorize("knowledge.pdf.search")
    assert first_ok is False
    assert first_reason == "tool_family_budget_exhausted:knowledge"
    assert budget.stop_new_tools is False

    second_ok, second_reason = budget.authorize("knowledge.pdf.search")
    assert second_ok is False
    assert second_reason == "tool_family_budget_exhausted:knowledge"
    assert budget.stop_new_tools is True
    assert budget.finalization_reason == "tool_family_budget_exhausted:knowledge"

    third_ok, third_reason = budget.authorize("web.search", entity="甲公司")
    assert third_ok is False
    assert third_reason == "tool_family_budget_exhausted:knowledge"


def test_market_family_allows_four_calls() -> None:
    budget = MainAgentBudgetController(started_monotonic=time.monotonic())
    for _ in range(4):
        assert budget.authorize("iwencai.finance.query")[0] is True
        budget.register("iwencai.finance.query")
    allowed, reason = budget.authorize("iwencai.finance.query")
    assert allowed is False
    assert reason == "tool_family_budget_exhausted:market"
    assert budget.stop_new_tools is False


@pytest.mark.asyncio
async def test_pdf_catalog_miss_returns_terminal_cross_family_fallback(monkeypatch) -> None:
    async def fake_execute_registered_tool(**kwargs):
        return ToolResult(
            tool_id=kwargs["tool_id"],
            ok=True,
            data={"documents": [], "count": 0},
        )

    monkeypatch.setattr(
        "agents.main_deep_agent.tools.factory.execute_registered_tool",
        fake_execute_registered_tool,
    )
    context = AgentRuntimeContext(
        permissions=("knowledge.pdf.catalog", "knowledge.pdf.search", "web.search"),
    )
    tools = build_main_tools(
        context=context,
        budget=MainAgentBudgetController(started_monotonic=time.monotonic()),
        journal=MainAgentProgressJournal(),
    )
    catalog = {item.name: item for item in tools}["catalog_pdf_knowledge_tool"]

    result = await catalog.ainvoke(
        {"entity": "未收录公司", "categories": ["annual_reports"]}
    )

    assert result["error"] == "local_document_not_found"
    assert result["retryable"] is False
    assert result["stop_same_tool"] is True
    assert result["fallback_tool_ids"] == ["web.search"]


@pytest.mark.asyncio
async def test_pdf_search_requires_doc_id_returned_by_catalog(monkeypatch) -> None:
    async def fake_execute_registered_tool(**kwargs):
        tool_id = kwargs["tool_id"]
        if tool_id == "knowledge.pdf.catalog":
            return ToolResult(
                tool_id=tool_id,
                ok=True,
                data={
                    "documents": [{"doc_id": "report-2025"}],
                    "count": 1,
                },
            )
        return ToolResult(tool_id=tool_id, ok=True, data={"evidence": [], "count": 0})

    monkeypatch.setattr(
        "agents.main_deep_agent.tools.factory.execute_registered_tool",
        fake_execute_registered_tool,
    )
    context = AgentRuntimeContext(
        permissions=("knowledge.pdf.catalog", "knowledge.pdf.search", "web.search"),
    )
    tools = build_main_tools(
        context=context,
        budget=MainAgentBudgetController(started_monotonic=time.monotonic()),
        journal=MainAgentProgressJournal(),
    )
    by_name = {item.name: item for item in tools}
    await by_name["catalog_pdf_knowledge_tool"].ainvoke(
        {"entity": "示例公司", "categories": ["annual_reports"]}
    )

    rejected = await by_name["search_pdf_knowledge_tool"].ainvoke(
        {
            "query": "营业收入",
            "categories": ["annual_reports"],
            "doc_ids": ["another-report"],
        }
    )
    autofilled = await by_name["search_pdf_knowledge_tool"].ainvoke(
        {
            "query": "营业收入",
            "categories": ["annual_reports"],
        }
    )
    admitted = await by_name["search_pdf_knowledge_tool"].ainvoke(
        {
            "query": "营业收入 显式 doc_ids",
            "categories": ["annual_reports"],
            "doc_ids": ["report-2025"],
        }
    )

    assert rejected["error"] == "pdf_doc_ids_not_cataloged"
    assert rejected["retryable"] is False
    assert rejected["requires_tool_id"] == "knowledge.pdf.catalog"
    assert autofilled["error"] == "local_document_evidence_not_found"
    # knowledge 族配额=2：catalog + 自动补齐 search 已用尽
    assert admitted["error"] == "tool_family_budget_exhausted:knowledge"
    assert admitted["fallback_tool_ids"] == ["web.search"]


def test_research_tool_cutoff_starts_finalization() -> None:
    budget = MainAgentBudgetController(
        started_monotonic=time.monotonic() - 40.5,
        budget_tier="deep_research",
        soft_seconds=42.0,
        hard_seconds=50.0,
    )
    allowed, reason = budget.authorize("web.search", entity="Apple")
    assert allowed is False
    assert reason == "research_tool_cutoff"
    assert budget.stop_new_tools is True
    assert budget.finalization_started_at is not None


def test_conflicts_distinguish_rounding_and_unresolved_scope() -> None:
    base_metadata = {
        "displayable": True,
        "entity": "Amazon",
        "source_grade": "secondary",
        "facts": [],
    }
    first = _evidence("e1", "web").model_copy(update={
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Amazon", "metric": "net_income_growth",
                "fiscal_period": "FY2025 Q3", "value": 38,
                "unit": "%", "currency": "",
            }],
        }
    })
    second = _evidence("e2", "web").model_copy(update={
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Amazon", "metric": "net_income_growth",
                "fiscal_period": "FY2025 Q3", "value": 39,
                "unit": "%", "currency": "",
            }],
        }
    })
    conflicts = detect_fact_conflicts([first, second])
    assert conflicts[0]["resolved"] is True
    assert conflicts[0]["resolution"] == "rounding_difference"

    eps_first = first.model_copy(update={
        "evidence_id": "eps1",
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Amazon", "metric": "eps_consensus",
                "fiscal_period": "FY2025 Q3", "value": 1.56,
                "unit": "currency/share", "currency": "USD",
            }],
        },
    })
    eps_second = second.model_copy(update={
        "evidence_id": "eps2",
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Amazon", "metric": "eps_consensus",
                "fiscal_period": "FY2025 Q3", "value": 1.58,
                "unit": "currency/share", "currency": "USD",
            }],
        },
    })
    eps_conflict = detect_fact_conflicts([eps_first, eps_second])[0]
    assert eps_conflict["resolved"] is False
    assert eps_conflict["resolution"] == "scope_difference"


@pytest.mark.asyncio
async def test_salvage_never_leaks_raw_web_json_or_investment_gap() -> None:
    raw = _evidence("raw", "web").model_copy(update={
        "content": '{"answer":"x","configured":true,"results":[]}',
        "metadata": {"source_family": "web", "displayable": False},
    })
    update = await main_evidence_quality_gate({
        "messages": [HumanMessage(content="苹果和亚马逊财报超预期，哪些业务增长最亮眼？")],
        "main_agent_response": None,
        "main_agent_journal": {"source_families": ["web"]},
        "evidence": [raw],
    })
    assert "configured" not in update["summary"]
    assert "买入" not in update["summary"]
    assert "results" not in update["summary"]
    assert "风险承受能力" not in update["summary"]
    assert update["main_quality_metrics"]["raw_json_leak"] is False


@pytest.mark.asyncio
async def test_apple_amazon_comparison_keeps_evidence_backed_facts() -> None:
    question = "苹果和亚马逊财报超预期，哪些业务增长最亮眼？"
    apple = web_evidence_from_payload({
        "answer": "内部聚合摘要",
        "results": [{
            "title": "Apple reports fiscal 2025 fourth quarter results",
            "url": "https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/",
            "content": "Apple fiscal 2025 fourth quarter Services revenue grew 15%.",
            "published_date": "2025-10-30",
        }],
    }, entity="Apple")[0]
    amazon = web_evidence_from_payload({
        "answer": "内部聚合摘要",
        "results": [{
            "title": "Amazon announces FY2025 Q4 results",
            "url": "https://www.aboutamazon.com/news/company-news/amazon-q4-2025-results",
            "content": "Amazon FY2025 Q4 AWS revenue growth was 24%.",
            "published_date": "2026-02-05",
        }],
    }, entity="Amazon")[0]
    response = MainAgentResponse(
        mode="grounded",
        gaps=["当前资料不足以判断是否买入。"],
        
        heading="业务亮点",
        statements=[
            MainAgentStatement(
                text="Apple 服务业务收入同比增长15%。",
                evidence_ids=[apple.evidence_id],
                entity="Apple",
                metric="services_revenue_growth",
                period="FY2025 Q4",
                unit="%",
            ),
            MainAgentStatement(
                text="Amazon AWS收入同比增长24%。",
                evidence_ids=[amazon.evidence_id],
                entity="Amazon",
                metric="aws_revenue_growth",
                period="FY2025 Q4",
                unit="%",
            ),
        ],
    )
    update = await main_evidence_quality_gate({
        "messages": [HumanMessage(content=question)],
        "main_agent_response": response,
        "main_agent_journal": {"source_families": ["web"]},
        "evidence": [apple, amazon],
    })
    assert update["execution_status"] == "completed_with_gaps"
    assert update["quality_report"].passed is True
    assert "15%" in update["summary"]
    assert "24%" in update["summary"]
    assert "configured" not in update["summary"]
    assert "买入" not in update["summary"]


@pytest.mark.asyncio
async def test_multi_currency_comparison_keeps_answer_and_adds_soft_gap() -> None:
    """暂无汇率时仍发布对比，仅软提示币种口径，不拒答。"""
    tencent = Evidence(
        evidence_id="main:tencent-hkd",
        task_id="main",
        source_type="iwencai.query",
        provider="market",
        title="腾讯营收",
        content="腾讯 2025 营收 8377 亿港元",
        metadata={
            "entity": "Tencent",
            "fiscal_period": "FY2025 Q4",
            "displayable": True,
            "display_text": "腾讯 2025 营收 8377 亿港元",
            "facts": [{
                "entity": "Tencent",
                "metric": "revenue_growth",
                "fiscal_period": "FY2025 Q4",
                "value": 8377,
                "unit": "亿港元",
                "currency": "HKD",
            }],
        },
    )
    alibaba = Evidence(
        evidence_id="main:alibaba-usd",
        task_id="main",
        source_type="iwencai.query",
        provider="market",
        title="阿里营收",
        content="阿里 FY2026 营收 1484 亿美元",
        metadata={
            "entity": "Alibaba",
            "fiscal_period": "FY2026 Q4",
            "displayable": True,
            "display_text": "阿里 FY2026 营收 1484 亿美元",
            "facts": [{
                "entity": "Alibaba",
                "metric": "revenue_growth",
                "fiscal_period": "FY2026 Q4",
                "value": 1484,
                "unit": "亿美元",
                "currency": "USD",
            }],
        },
    )
    response = MainAgentResponse(
        mode="grounded",
        
        heading="营收对照",
        statements=[
            MainAgentStatement(
                text="腾讯最近完整财年营收约 8377 亿港元。",
                evidence_ids=[tencent.evidence_id],
            ),
            MainAgentStatement(
                text="阿里最近完整财年营收约 1484 亿美元。",
                evidence_ids=[alibaba.evidence_id],
            ),
        ],
    )
    update = await main_evidence_quality_gate({
        "messages": [HumanMessage(content="比较腾讯和阿里营收")],
        "main_agent_response": response,
        "evidence": [tencent, alibaba],
    })
    assert update["quality_report"].passed is True
    assert "8377" in update["summary"]
    assert "1484" in update["summary"]
    assert "多种币种" in update["summary"]
    assert "非同一会计期间" in update["summary"] or "不同会计期间" in update["summary"]
    assert update["main_quality_metrics"]["multi_currency_comparison"] is True
    assert update["main_quality_metrics"]["multi_fiscal_period_comparison"] is True
    assert update["execution_status"] == "completed_with_gaps"


@pytest.mark.asyncio
async def test_cross_period_statement_is_removed() -> None:
    question = "苹果和亚马逊财报对比"
    apple = web_evidence_from_payload({
        "results": [{
            "title": "Apple FY2025 Q4 results",
            "url": "https://apple.com/newsroom/results",
            "content": "Apple FY2025 Q4 Services revenue grew 15%.",
            "published_date": "2025-10-30",
        }],
    }, entity="Apple")[0]
    amazon = web_evidence_from_payload({
        "results": [{
            "title": "Amazon FY2025 Q4 results",
            "url": "https://aboutamazon.com/results",
            "content": "Amazon FY2025 Q4 AWS revenue growth was 24%.",
            "published_date": "2026-02-05",
        }],
    }, entity="Amazon")[0]
    response = MainAgentResponse(
        mode="grounded",
        
        heading="比较",
        statements=[MainAgentStatement(
            text="Apple FY2025 Q3服务收入增长15%。",
            evidence_ids=[apple.evidence_id],
            entity="Apple",
            metric="services_revenue_growth",
            period="FY2025 Q3",
            unit="%",
        )],
    )
    update = await main_evidence_quality_gate({
        "messages": [HumanMessage(content=question)],
        "main_agent_response": response,
        "main_agent_journal": {"source_families": ["web"]},
        "evidence": [apple, amazon],
    })
    assert "FY2025 Q3服务收入增长15%" not in update["summary"]
    assert update["main_quality_metrics"]["cross_period_mismatch"] == 1


@pytest.mark.asyncio
async def test_tavily_applies_tool_side_domain_allowlist(monkeypatch) -> None:
    import tools.web_search as web_search_tool

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answer": "", "results": []}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(web_search_tool.settings, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(web_search_tool.settings, "WEB_SEARCH_ALLOWED_DOMAINS", "")
    monkeypatch.setattr(web_search_tool.settings, "WEB_SEARCH_MAX_DOMAINS", 20)
    monkeypatch.setattr(web_search_tool.httpx, "AsyncClient", FakeClient)

    await web_search_tool.search_web.ainvoke({"query": "今日 A 股热点有哪些"})
    assert captured["payload"]["include_domains"] == list(DEFAULT_WEB_SEARCH_DOMAINS)[:20]
    assert "eastmoney.com" in captured["payload"]["include_domains"]
    assert "gov.cn" in captured["payload"]["include_domains"]

    captured.clear()
    await web_search_tool.fetch_web_search("今日财经热搜", scope="open")
    assert "include_domains" not in captured["payload"]


def test_web_search_domain_policy_allowlist_or_open() -> None:
    domains = resolve_search_domains(scope="allowlist")
    assert domains == list(DEFAULT_WEB_SEARCH_DOMAINS)[:20]
    assert "eastmoney.com" in domains
    assert "gov.cn" in domains
    assert len(domains) == min(20, len(DEFAULT_WEB_SEARCH_DOMAINS))
    assert resolve_search_domains(scope="open") == []
    assert len(resolve_search_domains(scope="allowlist", max_domains=5)) == 5

def test_web_search_soft_score_filter_keeps_threshold_hits_in_order() -> None:
    from tools.web_search import _rank_and_filter_results

    kept, stats = _rank_and_filter_results(
        [
            {"title": "low", "url": "https://a", "content": "x", "score": 0.1},
            {"title": "mid", "url": "https://b", "content": "y", "score": 0.36},
            {"title": "high", "url": "https://c", "content": "z", "score": 0.5},
        ],
        min_score=0.35,
        max_results=4,
    )
    assert [item["title"] for item in kept] == ["high", "mid"]
    assert stats["raw_count"] == 3
    assert stats["kept_count"] == 2
    assert stats["dropped_by_score"] == 1
    assert stats["fallback_kept"] is False
    assert stats["max"] == 0.5


def test_web_search_score_filter_returns_empty_when_all_below_threshold() -> None:
    from tools.web_search import _rank_and_filter_results

    kept, stats = _rank_and_filter_results(
        [
            {"title": "a", "url": "https://a", "content": "x", "score": 0.05},
            {"title": "b", "url": "https://b", "content": "y", "score": 0.12},
            {"title": "c", "url": "https://c", "content": "z", "score": 0.08},
        ],
        min_score=0.35,
        max_results=4,
    )
    assert kept == []
    assert stats["fallback_kept"] is False
    assert stats["kept_count"] == 0
    assert stats["dropped_by_score"] == 3


def test_web_sanitize_payload_keeps_displayable_summaries_only() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import (
        _compact_evidence_for_model,
        _sanitize_tool_payload,
    )

    evidence = web_evidence_from_payload(
        {
            "results": [
                {
                    "title": "Tencent 2025 annual results",
                    "url": "https://www.tencent.com/en-us/investors/2025-results.html",
                    "content": (
                        "Tencent 2025 年报显示营业收入 8377 亿元，同比增长 13%。"
                        "增值服务与广告业务均录得增长。"
                    ),
                    "published_date": "2026-03-19",
                    "score": 0.8,
                },
                {
                    "title": "Unrelated sports podcast",
                    "url": "https://podcasts.example.com/show",
                    "content": "This episode talks about basketball and has no company filing.",
                    "published_date": "2026-03-18",
                    "score": 0.3,
                },
            ],
        },
        entity="Tencent",
    )
    assert len(evidence) >= 1
    # 强制第二条为不可展示，模拟实体错配。
    evidence[1].metadata["displayable"] = False
    evidence[1].metadata["entity_mismatch"] = True
    evidence[1].content = "x" * 1500

    sanitized = _sanitize_tool_payload(
        "web.search",
        {"answer": "聚合答案应被剥离", "results": [{"content": "raw"}]},
        evidence=evidence,
    )
    assert sanitized["answer_omitted"] is True
    assert "answer" not in sanitized
    assert sanitized["dropped_low_quality"] >= 1
    assert sanitized["displayable_fallback"] is False
    assert sanitized["results"]
    assert all("content" not in row for row in sanitized["results"])
    assert all(row.get("summary") for row in sanitized["results"])
    kept_ids = {item.evidence_id for item in evidence if item.metadata.get("displayable")}
    assert {row["evidence_id"] for row in sanitized["results"]} <= kept_ids

    compact = _compact_evidence_for_model(evidence, tool_id="web.search")
    assert compact
    assert all("content" not in item for item in compact)
    assert all(item["evidence_id"] for item in compact)
    assert all(item["metadata"].get("displayable") for item in compact)


def test_web_sanitize_payload_fallback_when_all_undisplayable() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import (
        _compact_evidence_for_model,
        _sanitize_tool_payload,
    )

    evidence = [
        Evidence(
            evidence_id="web:aaaaaaaaaaaaaaaaaaaa",
            task_id="main",
            source_type="web.search",
            provider="example.com",
            title="噪声页",
            content="很长的不可展示正文" * 40,
            url="https://example.com/noise",
            metadata={
                "displayable": False,
                "display_text": "噪声页摘要",
                "entity": "Other",
                "fiscal_period": "",
                "facts": [],
            },
        ),
        Evidence(
            evidence_id="web:bbbbbbbbbbbbbbbbbbbb",
            task_id="main",
            source_type="web.search",
            provider="example.com",
            title="另一噪声",
            content="另一段很长的正文" * 40,
            url="https://example.com/noise2",
            metadata={
                "displayable": False,
                "display_text": "另一噪声摘要",
                "entity": "Other",
                "fiscal_period": "",
                "facts": [],
            },
        ),
    ]
    sanitized = _sanitize_tool_payload("web.search", {"results": []}, evidence=evidence)
    assert sanitized["displayable_fallback"] is True
    assert len(sanitized["results"]) == 1
    assert sanitized["results"][0]["evidence_id"] == "web:aaaaaaaaaaaaaaaaaaaa"
    assert sanitized["results"][0]["summary"] == "噪声页摘要"
    compact = _compact_evidence_for_model(evidence, tool_id="web.search")
    assert len(compact) == 1
    assert compact[0]["evidence_id"] == "web:aaaaaaaaaaaaaaaaaaaa"
    assert "content" not in compact[0]


@pytest.mark.asyncio
async def test_inference_with_one_publisher_keeps_verified_facts() -> None:
    evidence = web_evidence_from_payload({
        "results": [{
            "title": "Tencent 2025 annual results",
            "url": "https://www.tencent.com/en-us/investors/2025-results.html",
            "content": "Tencent 2025 年报显示增值服务收入同比增长12%。",
            "published_date": "2026-03-19",
        }],
    }, entity="Tencent")[0]
    response = MainAgentResponse(
        mode="grounded",
        
        heading="近况",
        statements=[
            MainAgentStatement(
                text="腾讯增值服务收入同比增长12%。",
                statement_type="fact",
                evidence_ids=[evidence.evidence_id],
            ),
            MainAgentStatement(
                text="若行业景气延续，后续仍值得跟踪。",
                statement_type="inference",
                evidence_ids=[evidence.evidence_id],
            ),
        ],
    )
    update = await main_evidence_quality_gate({
        "messages": [HumanMessage(content="腾讯最近状况")],
        "main_agent_response": response,
        "evidence": [evidence],
    })
    assert "增值服务收入同比增长12%" in update["summary"]
    assert "第二个独立来源" in update["summary"]
    assert update["execution_status"] == "completed_with_gaps"
    assert update["quality_report"].passed is True


def test_disclaimer_only_official_page_has_no_official_earnings() -> None:
    evidence = web_evidence_from_payload({
        "results": [{
            "title": "Apple reports fourth quarter results",
            "url": "https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/",
            "content": (
                "This press release contains forward-looking statements within the "
                "meaning of the Private Securities Litigation Reform Act. Actual "
                "results could differ materially."
            ),
            "published_date": "2025-10-30",
        }],
    }, entity="Apple")[0]
    assert evidence.metadata["extraction_failed"] is True
    assert "official_earnings" not in evidence.metadata["available_fields"]
    assert evidence.metadata["source_grade"] == "secondary"


def test_entity_mismatch_podcast_is_not_counted_as_apple() -> None:
    evidence = web_evidence_from_payload({
        "results": [{
            "title": "Norbert's Wealth Dome",
            "url": "https://podcast.example.com/episode-1",
            "content": "Visa net income $20B +11%. Salesforce and Snowflake also reported.",
            "published_date": "2026-07-01",
        }],
    }, entity="Apple")[0]
    assert evidence.metadata["entity_mismatch"] is True
    assert evidence.metadata["displayable"] is False
    assert evidence.metadata["available_fields"] == []


def test_web_results_must_match_one_requested_entity() -> None:
    evidence = web_evidence_from_payload(
        {
            "results": [
                {
                    "title": "腾讯控股发布年度业绩",
                    "url": "https://example.com/tencent",
                    "content": "腾讯控股2025财年营业收入与净利润。",
                },
                {
                    "title": "无关公司年度报告",
                    "url": "https://example.com/unrelated",
                    "content": "龙芯中科2025年营业收入与净利润。",
                },
            ]
        },
        entity="",
        entities=["腾讯控股", "阿里巴巴"],
    )
    assert evidence[0].metadata["entity"] == "腾讯控股"
    assert evidence[0].metadata["displayable"] is True
    assert evidence[1].metadata["entity_mismatch"] is True
    assert evidence[1].metadata["displayable"] is False


def test_title_platform_and_consumer_subdomain_do_not_count_as_official_entity() -> None:
    """正文未提目标公司时，标题平台名 + 消费子域不得记作该公司官方证据。"""
    evidence = web_evidence_from_payload({
        "results": [{
            "title": "Beta Finch - Healthcare & Devices - EN - Podcast - Apple Podcasts",
            "url": "https://podcasts.apple.com/us/podcast/beta-finch/id123",
            "content": (
                "Cigna Healthcare really outperformed. Evernorth earnings were slightly ahead, "
                "while new revenue guidance is $47.4 to $48.1 billion."
            ),
        }],
    }, entity="Apple")[0]
    assert evidence.metadata["entity_mismatch"] is True
    assert evidence.metadata["displayable"] is False
    assert evidence.metadata["source_grade"] == "secondary"


def test_english_month_date_sets_published_at() -> None:
    evidence = web_evidence_from_payload({
        "results": [{
            "title": "Amazon.com Announces Second Quarter Results",
            "url": "https://www.aboutamazon.com/news/company-news/amazon-q2-2026-earnings",
            "content": (
                "SEATTLE—(BUSINESS WIRE) July 30, 2026—Amazon.com, Inc. today announced "
                "financial results for its second quarter ended June 30, 2026. "
                "AWS net sales increased 37% year-over-year to $42.2 billion."
            ),
        }],
    }, entity="Amazon")[0]
    assert evidence.published_at == "2026-07-30"
    assert evidence.metadata["fiscal_period"] == "FY2026 Q2"
    assert "official_earnings" in evidence.metadata["available_fields"]
    assert len(evidence.metadata["display_text"]) < 400


def test_empty_fiscal_period_conflict_does_not_blacklist_evidence() -> None:
    base_metadata = {
        "displayable": True,
        "entity": "Tencent",
        "source_grade": "secondary",
        "facts": [],
    }
    first = _evidence("e1", "web").model_copy(update={
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Tencent", "metric": "revenue_growth",
                "fiscal_period": "", "value": 12,
                "unit": "%", "currency": "",
            }],
        }
    })
    second = _evidence("e2", "web").model_copy(update={
        "metadata": {
            **base_metadata,
            "facts": [{
                "entity": "Tencent", "metric": "revenue_growth",
                "fiscal_period": "", "value": 18,
                "unit": "%", "currency": "",
            }],
        }
    })
    conflicts = detect_fact_conflicts([first, second])
    assert conflicts[0]["resolution"] == "period_unknown"
    assert conflicts[0]["resolved"] is True
    assert conflicts[0]["evidence_ids"] == []


def test_finance_fact_lookup_builds_structured_evidence() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import (
        _structured_tool_evidence,
    )

    evidence = _structured_tool_evidence(
        "knowledge.fact.lookup",
        {
            "ok": True,
            "answer": "腾讯 2025 年营业收入 6600 亿元",
            "facts": [{
                "company": "腾讯",
                "period_year": 2025,
                "metric": "营业收入",
                "value": "6600",
                "unit": "亿元",
                "currency": "CNY",
            }],
        },
    )[0]
    assert evidence.metadata["source_grade"] == "structured"
    assert "official_earnings" in evidence.metadata["available_fields"]
    assert evidence.metadata["entity"] == "腾讯"
    assert evidence.metadata["displayable"] is True

    rating = _structured_tool_evidence(
        "iwencai.rating.query",
        {
            "ok": True,
            "answer": "Apple FY2026 Q3 EPS consensus $1.56，实际业绩超预期",
            "entity": "Apple",
            "fiscal_period": "FY2026 Q3",
        },
    )[0]
    assert "expectation_comparison" in rating.metadata["available_fields"]


@pytest.mark.parametrize(
    "tool_id,payload",
    [
        ("knowledge.faq.search", {"query": "市盈率", "evidence": []}),
        ("knowledge.pdf.search", {"query": "营业收入", "evidence": [], "count": 0}),
    ],
)

def test_empty_local_retrieval_does_not_fabricate_evidence(tool_id, payload) -> None:
    from agents.main_deep_agent.tools.evidence_adapter import _evidence_from_payload

    assert _evidence_from_payload(tool_id, payload) == []


@pytest.mark.asyncio
async def test_summary_rejects_empty_findings_when_evidence_present() -> None:
    from agents.research_workflow.contracts import ResearchContextSummary
    from agents.research_workflow.deep_agent.context_middleware import (
        GovernedResearchSummarizationMiddleware,
    )

    class FakeModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return ResearchContextSummary(
                objective="研究 Astra 传闻",
                findings=[],
                evidence_ids=["web:abc123"],
            )

    journal = MainAgentProgressJournal()
    journal.evidence["web:abc123"] = Evidence(
        evidence_id="web:abc123",
        task_id="main",
        source_type="web.search",
        provider="example.com",
        title="Astra 报道",
        content="旧文提到 Astra",
        metadata={
            "displayable": True,
            "display_text": "旧文提到 Astra",
            "facts": [],
        },
    )
    middleware = GovernedResearchSummarizationMiddleware(FakeModel(), journal=journal)
    rendered = await middleware._acreate_summary([
        SimpleNamespace(
            type="tool",
            content='{"ok": true, "evidence_id": "web:abc123", "text": "旧文提到 Astra"}',
        )
    ])
    # 占位 JSON 之后还会追加确定性证据索引，只解析首个 JSON 对象。
    payload = json.loads(rendered.split("\n\n[确定性证据索引", 1)[0])
    assert payload["failed_sources"] == ["summarization_empty_findings"]
    assert payload["evidence_ids"] == ["web:abc123"]
    assert payload["findings"]
    assert payload["findings"][0]["evidence_ids"] == ["web:abc123"]
    assert "确定性证据索引" in rendered
    assert "web:abc123" in rendered


def test_iwencai_query_builds_displayable_structured_evidence() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import _evidence_from_payload

    evidence = _evidence_from_payload(
        "iwencai.query",
        {
            "ok": True,
            "provider": "iwencai",
            "query": "腾讯近两年营业收入净利润毛利率",
            "answer": "腾讯财报表现最好，建议继续关注。",
            "data": {
                "datas": [{
                    "股票简称": "腾讯控股",
                    "营业收入": "837768030400",
                    "归母净利润": "250563924800",
                    "销售毛利率": "56.21%",
                    "period_year": 2025,
                }],
                "facts": [{
                    "company": "腾讯",
                    "period_year": 2025,
                    "metric": "营业收入",
                    "value": "837768030400",
                    "unit": "人民币",
                    "currency": "CNY",
                }],
            },
        },
    )
    assert len(evidence) == 1
    item = evidence[0]
    assert item.evidence_id.startswith("main:")
    assert item.metadata["displayable"] is True
    assert item.metadata["source_grade"] == "structured"
    assert item.metadata["display_text"]
    assert item.metadata["facts"]
    assert item.metadata["facts"][0]["currency"] == "CNY"
    assert item.metadata["fiscal_period"] == "FY2025 Q4"
    preview = item.metadata["preview_table"]
    assert "营业收入" in preview["columns"]
    assert preview["rows"][0][preview["columns"].index("营业收入")] == "837768030400"
    assert "=" not in item.metadata["display_text"]
    assert "建议继续关注" not in item.content


def test_iwencai_model_answer_without_structured_rows_is_not_evidence() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import _evidence_from_payload

    evidence = _evidence_from_payload(
        "iwencai.query",
        {
            "ok": True,
            "provider": "iwencai",
            "query": "比较三家公司财务数据",
            "data": {"answer": "腾讯规模最大，美团仍需看拐点。", "datas": []},
        },
    )
    assert evidence == []


def test_iwencai_multi_entity_rows_become_separate_evidence() -> None:
    from agents.main_deep_agent.tools.evidence_adapter import _evidence_from_payload

    evidence = _evidence_from_payload(
        "iwencai.query",
        {
            "ok": True,
            "provider": "iwencai",
            "query": "腾讯控股 阿里巴巴 美团 2025财年营业收入",
            "data": {
                "datas": [
                    {"股票代码": "0700.HK", "营业收入": "8377.68亿", "报告期截止日": 20251231},
                    {"股票代码": "9988.HK", "营业收入": "9963.47亿", "报告期截止日": 20250331},
                    {"股票代码": "3690.HK", "营业收入": "4065.94亿", "报告期截止日": 20251231},
                ]
            },
        },
    )
    assert len(evidence) == 3
    assert all(item.metadata["entity"] == "" for item in evidence)
    first_preview = evidence[0].metadata["preview_table"]
    date_index = first_preview["columns"].index("报告期截止日")
    assert first_preview["rows"][0][date_index] == "20251231"


def test_preview_article_is_not_official_earnings() -> None:
    evidence = web_evidence_from_payload(
        {
            "results": [{
                "title": "甲公司2026Q2业绩前瞻：营收有望同比增12%",
                "url": "https://research.example.com/preview/jia",
                "content": (
                    "业绩前瞻显示，甲公司即将于8月发布财报。"
                    "一致预期营收同比增长12%，广告业务预计提速；正式数据待公布。"
                ),
                "published_date": "2026-07-20",
            }],
        },
        entity="甲公司",
    )[0]
    assert evidence.metadata["is_preview"] is True
    assert "official_earnings" not in evidence.metadata["available_fields"]
    assert "earnings_preview" in evidence.metadata["available_fields"]
    assert evidence.metadata["display_text"].startswith("【未披露·前瞻/预估】")


def test_public_tool_step_detail_uses_displayable_evidence_only() -> None:
    evidence = _evidence("e-detail", "financial")
    evidence.metadata.update(
        {
            "displayable": True,
            "display_text": "川金诺（300505.SZ），归母净利润同比增长率[20250930] 175.61",
            "entity": "川金诺",
            "fiscal_period": "FY2025 Q3",
            "preview_table": {
                "query": "川金诺2025年前三季度归母净利润同比增长率",
                "columns": ["股票代码", "股票简称", "归母净利润同比增长率[20250930]"],
                "rows": [["300505.SZ", "川金诺", "175.61"]],
            },
            "facts": [
                {
                    "entity": "川金诺",
                    "metric": "net_income_growth",
                    "fiscal_period": "FY2025 Q3",
                    "value": 175.61,
                    "unit": "%",
                }
            ],
        }
    )
    running = build_public_tool_step_detail(
        tool_id="iwencai.query",
        source_family="market",
        query="川金诺2025年前三季度净利润同比增长多少",
        status="running",
    )
    assert running is not None
    assert running["title"] == "问财数据"
    assert running["query"].startswith("川金诺")
    assert "display_text" not in running

    done = build_public_tool_step_detail(
        tool_id="iwencai.query",
        source_family="market",
        query="川金诺2025年前三季度净利润同比增长多少",
        evidence=[evidence],
        status="done",
    )
    assert done is not None
    assert done["title"] == "问财数据"
    assert "175.61" in done["display_text"]
    assert "股票代码=" not in done["display_text"]
    assert done["columns"][0] == "股票代码"
    assert done["rows"][0][1] == "川金诺"


def test_sanitize_follow_ups_keeps_model_text_and_filters_action_language() -> None:
    follow_ups = sanitize_follow_ups(
        [
            "川金诺同期营业收入是多少？",
            "川金诺2025年前三季度净利润同比增长多少？",  # 与当前问题重复
            "川金诺现在适合买入吗？",
            "川金诺同期毛利率是多少？",
        ],
        query="川金诺2025年前三季度净利润同比增长多少？",
        investment_action_sensitive=True,
    )
    assert follow_ups == [
        "川金诺同期营业收入是多少？",
        "川金诺同期毛利率是多少？",
    ]


def test_build_answer_charts_requires_multi_period_facts() -> None:
    single = _evidence("e-single", "financial")
    single.metadata.update(
        {
            "entity": "川金诺",
            "facts": [
                {
                    "entity": "川金诺",
                    "metric": "net_income_growth",
                    "fiscal_period": "FY2025 Q3",
                    "value": 175.61,
                    "unit": "%",
                }
            ],
        }
    )
    assert build_answer_charts(evidence=[single]) == []

    multi = _evidence("e-multi", "financial")
    multi.metadata.update(
        {
            "entity": "川金诺",
            "facts": [
                {
                    "entity": "川金诺",
                    "metric": "net_income_growth",
                    "fiscal_period": "FY2023 Q4",
                    "value": -40.0,
                    "unit": "%",
                },
                {
                    "entity": "川金诺",
                    "metric": "net_income_growth",
                    "fiscal_period": "FY2024 Q4",
                    "value": 80.0,
                    "unit": "%",
                },
                {
                    "entity": "川金诺",
                    "metric": "net_income_growth",
                    "fiscal_period": "FY2025 Q3",
                    "value": 175.61,
                    "unit": "%",
                },
            ],
        }
    )
    charts = build_answer_charts(evidence=[multi])
    assert len(charts) == 1
    assert charts[0]["categories"] == ["FY2023 Q4", "FY2024 Q4", "FY2025 Q3"]
    assert charts[0]["lines"][0]["values"] == [-40.0, 80.0, 175.61]


def test_sanitize_step_detail_drops_raw_json_and_extra_keys() -> None:
    cleaned = sanitize_step_detail(
        {
            "title": "财务数据",
            "query": "测试查询",
            "display_text": '{"ok": true, "secret": 1}',
            "raw_payload": {"datas": [{"a": 1}]},
            "columns": ["实体", "数值"],
            "rows": [["川金诺", "3.04亿"], ["坏行"]],
        }
    )
    assert cleaned is not None
    assert cleaned["title"] == "财务数据"
    assert cleaned["query"] == "测试查询"
    assert "display_text" not in cleaned
    assert "raw_payload" not in cleaned
    assert cleaned["rows"] == [["川金诺", "3.04亿"]]

    event = build_step_event(
        step_id="main-tool-1-iwencai.query",
        label="已取得财务数据",
        status="done",
        category="financial",
        short_label="财务数据",
        detail=cleaned,
    )
    assert event["type"] == "step"
    assert event["detail"]["query"] == "测试查询"


def test_tool_error_is_visible_in_step_detail() -> None:
    detail = build_public_tool_step_detail(
        tool_id="iwencai.query",
        source_family="market",
        query="腾讯控股 2025 财年营业收入",
        status="error",
        error="iwencai_timeout",
    )
    assert detail is not None
    assert detail["error"] == "iwencai_timeout"
    cleaned = sanitize_step_detail(detail)
    assert cleaned is not None
    assert cleaned["error"] == "iwencai_timeout"

