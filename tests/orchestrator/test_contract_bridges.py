"""Finance Worker、Evidence 与 v2 契约的边界测试。"""

import pytest

from agents.orchestrator.adapters import (
    agent_result_from_task_result,
    citation_from_evidence,
    evidence_from_citation,
)
from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    DocumentHitSet,
    DomainPlanningScope,
    Evidence,
    MarketFilter,
    MarketQueryPlan,
    MarketSort,
    TaskSpec,
)
from agents.finance_agent.workers.isolation import project_worker_updates_to_parent
from agents.orchestrator.agent_registry import _merge_output_evidence, invoke_agent
def test_task_result_converts_to_agent_result() -> None:
    worker_result = {
        "sub_task_id": "task-1",
        "question": "查询新能源行业",
        "type": "web_search",
        "context": "公开信息摘要",
        "coverage": "covered",
        "confidence": 0.8,
        "citations": [
            {
                "source": "公告页面",
                "url": "https://example.com/a",
                "snippet": "公告摘要",
                "source_type": "web",
            }
        ],
    }

    result = agent_result_from_task_result(worker_result)

    assert result.status == "completed"
    assert result.evidence[0].url == "https://example.com/a"
    assert result.metadata["worker_type"] == "web_search"


def test_evidence_and_citation_preserve_public_fields() -> None:
    citation = {
        "source": "同花顺问财",
        "snippet": "筛选结果",
        "source_type": "iwencai",
        "sub_task_id": "screen",
        "url": "https://example.com/query",
        "published_at": "2026-01-01",
        "confidence": 0.9,
    }

    evidence = evidence_from_citation(citation)
    restored = citation_from_evidence(evidence)

    assert evidence.task_id == "screen"
    assert evidence.source_type == "iwencai"
    assert evidence.confidence == 0.9
    assert restored["url"] == citation["url"]
    assert restored["published_at"] == citation["published_at"]


def test_agent_result_can_be_serialized() -> None:
    result = AgentResult(
        task_id="task-1",
        agent_id="agent-1",
        status="partial",
        evidence=[Evidence(evidence_id="e-1", source_type="faq")],
    )

    payload = result.model_dump_json()
    restored = AgentResult.model_validate_json(payload)

    assert restored.task_id == result.task_id
    assert restored.evidence[0].evidence_id == "e-1"


def test_market_contracts_can_be_serialized() -> None:
    plan = MarketQueryPlan(
        universe="A股",
        filters=[MarketFilter(field="pe_ttm", operator="lt", value=30)],
        enrichments=["filings", "institution_rating"],
        sort=[MarketSort(field="revenue_growth", direction="desc")],
        limit=10,
    )
    candidates = CandidateSet(
        dataset_id="candidate-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[{"symbol": "300750.SZ", "pe_ttm": 20.1}],
        query_plan=plan,
    )

    restored = CandidateSet.model_validate_json(candidates.model_dump_json())

    assert restored.query_plan is not None
    assert restored.query_plan.filters[0].field == "pe_ttm"
    assert restored.rows[0]["symbol"] == "300750.SZ"


def test_document_hit_set_can_be_serialized() -> None:
    documents = DocumentHitSet(
        query="某公司最新研报",
        channel="report",
        documents=[
            {
                "document_id": "r-1",
                "title": "公司深度报告",
                "summary": "摘要",
                "published_at": "2026-07-27",
                "organization": "某券商",
                "rating": "买入",
                "target_price": 100,
                "url": "https://example.test/report",
            }
        ],
        evidence_ids=["iwencai.report.search:q"],
    )

    restored = DocumentHitSet.model_validate_json(documents.model_dump_json())

    assert restored.documents[0].organization == "某券商"
    assert restored.documents[0].target_price == 100


@pytest.mark.asyncio
async def test_invoke_agent_passes_complete_dependency_state(monkeypatch) -> None:
    captured: dict = {}

    async def fake_stock_agent(state, *, query, config=None, runtime=None):
        captured["state"] = state
        captured["query"] = query
        return AgentResult(
            task_id="screen",
            agent_id="stock_screening_agent",
            status="completed",
        )

    monkeypatch.setattr(
        "agents.stock_screening_agent.run_stock_screening_agent",
        fake_stock_agent,
    )
    candidates = CandidateSet(
        dataset_id="candidate-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[{"symbol": "300750.SZ"}],
    )
    dependency = AgentResult(
        task_id="upstream",
        agent_id="stock_screening_agent",
        status="completed",
        answer="候选股票",
        structured_data=candidates.model_dump(),
        evidence=[Evidence(evidence_id="e-1", source_type="iwencai")],
    )

    await invoke_agent(
        TaskSpec(
            task_id="screen-next",
            objective="继续过滤候选股票",
            agent_id="stock_screening_agent",
            input_data={"top_k": 5},
        ),
        dependency_results=[dependency],
        memory_context={"default_market": "US"},
    )

    assert captured["query"] == "继续过滤候选股票"
    forwarded = captured["state"]["dependency_results"][0]
    assert forwarded.structured_data["dataset_id"] == "candidate-1"
    assert forwarded.evidence[0].evidence_id == "e-1"
    assert captured["state"]["task_input"] == {"top_k": 5}
    assert captured["state"]["memory_context"] == {"default_market": "US"}
    assert "task_memory_context" not in captured["state"]


@pytest.mark.asyncio
async def test_invoke_finance_agent_preserves_domain_scope(monkeypatch) -> None:
    captured: dict = {}

    class FakeFinanceAgent:
        async def ainvoke(self, state, config=None):
            del config
            captured.update(state)
            return {"summary": "ok"}

    import importlib

    finance_module = importlib.import_module("agents.finance_agent")
    monkeypatch.setattr(
        finance_module,
        "finance_agent",
        FakeFinanceAgent(),
        raising=False,
    )
    scope = DomainPlanningScope(
        parent_task_id="finance",
        parent_logical_task_id="finance",
        allowed_capabilities=["financial_query"],
        allowed_intents=["structured_metric"],
    )

    await invoke_agent(
        TaskSpec(
            task_id="finance",
            objective="查询营收",
            agent_id="finance_agent",
            input_data={
                "domain_planning_scope": scope.model_dump(mode="json"),
            },
        ),
        dependency_results=[],
    )

    assert captured["domain_planning_scope"]["parent_task_id"] == "finance"
    assert captured["task_input"]["domain_planning_scope"] == captured[
        "domain_planning_scope"
    ]


def test_finance_worker_projection_emits_unified_agent_result() -> None:
    updates = project_worker_updates_to_parent(
        {
            "task_results": [
                {
                    "sub_task_id": "financial-1",
                    "question": "查询营收",
                    "type": "financial_query",
                    "context": "营收保持增长",
                    "coverage": "covered",
                    "citations": [],
                }
            ],
            "financial_query_sql": "select ...",
        }
    )

    assert "financial_query_sql" not in updates
    assert len(updates["agent_results"]) == 1
    result = updates["agent_results"][0]
    assert result.task_id == "financial-1"
    assert result.agent_id == "finance_agent"
    assert result.status == "completed"


def test_root_registry_merges_finance_top_level_citations() -> None:
    result = AgentResult(
        task_id="worker-1",
        agent_id="finance_agent",
        status="completed",
        answer="财务数据摘要",
    )
    merged = _merge_output_evidence(
        "finance",
        [result],
        [
            {
                "source": "Annual_Report.pdf",
                "source_type": "pdf",
                "doc_id": "PDF-AR-01",
                "snippet": "营业收入 100 万元",
            }
        ],
    )

    assert merged.task_id == "finance"
    assert merged.evidence[0].metadata["doc_id"] == "PDF-AR-01"
