"""旧 Worker 数据与统一契约之间的转换测试。"""

from agents.orchestrator.adapters import (
    agent_result_from_task_result,
    citation_from_evidence,
    evidence_from_citation,
    subtask_from_task_spec,
    task_result_from_agent_result,
    task_spec_from_subtask,
)
from agents.orchestrator.contracts import AgentResult, Evidence, TaskSpec
from agents.finance_agent.workers.isolation import project_worker_updates_to_parent
from agents.orchestrator.agent_registry import _merge_output_evidence
from app.shared import SubTask


def test_subtask_and_task_spec_round_trip() -> None:
    legacy = SubTask(
        id="task-1",
        question="查询新能源行业",
        intent="market_event",
        reason="需要公开信息",
        type="web_search",
        evidence_chain=["web_search"],
    )

    converted = task_spec_from_subtask(legacy)
    restored = subtask_from_task_spec(converted)

    assert converted.task_id == "task-1"
    assert converted.required_capabilities == ["web_search"]
    assert restored.question == legacy.question
    assert restored.evidence_chain == legacy.evidence_chain


def test_task_result_and_agent_result_round_trip() -> None:
    legacy = {
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

    result = agent_result_from_task_result(legacy)
    restored = task_result_from_agent_result(result)

    assert result.status == "completed"
    assert result.evidence[0].url == "https://example.com/a"
    assert restored["coverage"] == "covered"
    assert restored["citations"][0]["source_type"] == "web"


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
