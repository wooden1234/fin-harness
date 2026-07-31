"""v2 来源工作流、确定性执行器和推理 Agent 边界测试。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.research_workflow.deep_agent.runtime import run_deep_research_agent
from agents.market_acquisition_workflow import run_market_acquisition_workflow
from agents.market_compute.executor import run_market_compute
from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    DeepResearchReport,
    Evidence,
    MarketFilter,
    MarketQueryPlan,
)
from agents.research_retrieval_workflow import run_research_retrieval_workflow
from agents.stock_screening_agent.node import run_stock_screening_agent
from tools.base import ToolResult


async def test_market_acquisition_requires_explicit_tool() -> None:
    result = await run_market_acquisition_workflow(
        {"task_input": {}},
        query="筛选新能源股票",
    )

    assert result.status == "clarify"
    assert result.error_code == "market_source_missing"


async def test_market_acquisition_rejects_research_tool() -> None:
    result = await run_market_acquisition_workflow(
        {
            "task_input": {
                "market_tool_id": "iwencai.report.search",
            }
        },
        query="查询研报",
    )

    assert result.status == "failed"
    assert result.error_code == "market_acquisition_tool_not_allowed"


async def test_research_workflow_rejects_market_tool() -> None:
    result = await run_research_retrieval_workflow(
        {
            "task_input": {
                "research_tool_id": "iwencai.market.query",
            }
        },
        query="查询行情",
    )

    assert result.status == "failed"
    assert result.error_code == "research_tool_not_allowed"


async def test_stock_screening_agent_owns_candidate_set_normalization(
    monkeypatch,
) -> None:
    async def fake_run(state, *, query, config=None, runtime=None):
        candidates = CandidateSet(
            dataset_id="screening-1",
            universe="A股",
            provider="iwencai",
            as_of="2026-07-27",
            rows=[{"code": "AAA", "pe_ttm": 20}],
            metadata={"producer_id": "stock_screening_agent"},
        )
        return AgentResult(
            task_id="stock-screening",
            agent_id="stock_screening_agent",
            status="completed",
            structured_data=candidates.model_dump(),
        )

    monkeypatch.setattr(
        "agents.stock_screening_agent.node.run_stock_screening_workflow",
        fake_run,
    )
    result = await run_stock_screening_agent({}, query="筛选低估值股票")
    candidates = CandidateSet.model_validate(result.structured_data)

    assert result.agent_id == "stock_screening_agent"
    assert candidates.rows == [{"code": "AAA", "pe_ttm": 20}]
    assert candidates.metadata["producer_id"] == "stock_screening_agent"


async def test_market_acquisition_uses_governed_source_tool(monkeypatch) -> None:
    async def fake_execute(tool_id, context, *, arguments, allowed_tool_ids):
        return ToolResult(
            tool_id=tool_id,
            ok=True,
            data={"data": {"rows": [{"code": "AAA", "price": 10}]}},
        )

    monkeypatch.setattr(
        "agents.iwencai_source_runtime.load_all_tools",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.iwencai_source_runtime.validate_tool_ids",
        lambda _: None,
    )
    monkeypatch.setattr(
        "agents.iwencai_source_runtime.execute_tool",
        fake_execute,
    )
    result = await run_market_acquisition_workflow(
        {
            "task_input": {
                "market_tool_id": "iwencai.market.query",
            }
        },
        query="查询 AAA 行情",
    )

    assert result.status == "completed"
    assert result.agent_id == "market_acquisition_workflow"
    assert result.metadata["tool_id"] == "iwencai.market.query"


async def test_research_workflow_normalizes_report_documents(monkeypatch) -> None:
    async def fake_execute(tool_id, context, *, arguments, allowed_tool_ids):
        return ToolResult(
            tool_id=tool_id,
            ok=True,
            data={
                "data": {
                    "results": [
                        {
                            "id": "r-1",
                            "title": "公司研究报告",
                            "organization": "某券商",
                            "rating": "增持",
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(
        "agents.iwencai_source_runtime.load_all_tools",
        lambda: None,
    )
    monkeypatch.setattr(
        "agents.iwencai_source_runtime.validate_tool_ids",
        lambda _: None,
    )
    monkeypatch.setattr(
        "agents.iwencai_source_runtime.execute_tool",
        fake_execute,
    )
    result = await run_research_retrieval_workflow(
        {
            "task_input": {
                "research_tool_id": "iwencai.report.search",
            }
        },
        query="查询公司研报",
    )

    assert result.status == "completed"
    assert result.structured_data["channel"] == "report"
    assert result.structured_data["documents"][0]["organization"] == "某券商"


async def test_research_workflow_converges_empty_result_to_uncovered(monkeypatch) -> None:
    async def fake_execute(tool_id, context, *, arguments, allowed_tool_ids):
        return ToolResult(tool_id=tool_id, ok=True, data={"data": {"results": []}})

    monkeypatch.setattr("agents.iwencai_source_runtime.load_all_tools", lambda: None)
    monkeypatch.setattr("agents.iwencai_source_runtime.validate_tool_ids", lambda _: None)
    monkeypatch.setattr("agents.iwencai_source_runtime.execute_tool", fake_execute)

    result = await run_research_retrieval_workflow(
        {"task_input": {"research_tool_id": "iwencai.report.search"}},
        query="查询不存在的公司研报",
    )

    assert result.status == "uncovered"
    assert result.error_code == "iwencai.report.search_empty_result"
    assert result.evidence[0].source_type == "iwencai.report.search"


async def test_market_compute_only_uses_upstream_candidate_set() -> None:
    source = CandidateSet(
        dataset_id="source-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[
            {"code": "AAA", "pe_ttm": 20},
            {"code": "BBB", "pe_ttm": 40},
        ],
    )
    result = await run_market_compute(
        {
            "dependency_results": [
                AgentResult(
                    task_id="market_acquire",
                    agent_id="stock_screening_agent",
                    status="completed",
                    structured_data=source.model_dump(),
                    evidence=[
                        Evidence(evidence_id="e-1", source_type="iwencai")
                    ],
                )
            ],
            "task_input": {
                "market_query_plan": MarketQueryPlan(
                    universe="A股",
                    filters=[
                        MarketFilter(field="pe_ttm", operator="lt", value=30)
                    ],
                ).model_dump()
            },
        },
        query="保留市盈率低于30的股票",
    )

    computed = CandidateSet.model_validate(result.structured_data)
    assert result.status == "completed"
    assert result.metadata["operation"] == "market.compute"
    assert [row["code"] for row in computed.rows] == ["AAA"]
    assert result.evidence[0].evidence_id == "e-1"


async def test_deep_research_rejects_tool_call_placeholder_as_evidence(monkeypatch) -> None:
    class FakeDeepAgent:
        async def ainvoke(self, state, config=None):
            return {"messages": [AIMessage(content="基于公告和研报形成研究结论")]}

    def fake_build(*, collector, **kwargs):
        collector.append(
            {
                "tool_id": "iwencai.report.search",
                "ok": True,
                "data": {"results": [{"title": "测试研报"}]},
                "error": "",
            }
        )
        return FakeDeepAgent()

    monkeypatch.setattr(
        "agents.research_workflow.deep_agent.runtime.build_deep_research_agent",
        fake_build,
    )
    result = await run_deep_research_agent(
        {"messages": []},
        query="深度研究宁德时代",
    )

    assert result.status == "failed"
    assert result.error_code == "deep_research_evidence_missing"
    assert result.metadata["runtime"] == "deep_agent"
    assert result.evidence == []
