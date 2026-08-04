"""V2 统一 Evidence 到最终 Citation 的边界测试。"""

from __future__ import annotations

from agents.final_answer.node import _filter_current_turn_citations
from agents.orchestrator.contracts import AgentResult, Evidence, TaskPlan, TaskSpec


def test_v2_citations_accept_research_tool_sources() -> None:
    result = AgentResult(
        task_id="research",
        agent_id="finance_agent",
        status="completed",
        evidence=[
            Evidence(
                evidence_id="e-1",
                task_id="research:report",
                source_type="iwencai.report.search",
            )
        ],
    )
    citations = [
        {
            "source": "公司研报",
            "snippet": "研报摘要",
            "source_type": "iwencai.report.search",
            "sub_task_id": "research:report",
        }
    ]

    filtered = _filter_current_turn_citations(
        {
            "task_plan": TaskPlan(
                plan_id="p1",
                query="研究公司",
                tasks=[
                    TaskSpec(
                        task_id="research",
                        objective="研究公司",
                        agent_id="finance_agent",
                    )
                ],
            ),
            "agent_results": [result],
            "evidence": result.evidence,
        },
        citations,
    )

    assert filtered == citations


def test_v2_citations_drop_unknown_source_without_provenance() -> None:
    filtered = _filter_current_turn_citations(
        {
            "agent_results": [
                AgentResult(
                    task_id="research",
                    agent_id="finance_agent",
                    status="completed",
                )
            ]
        },
        [
            {
                "source_type": "unknown",
                "sub_task_id": "research",
            }
        ],
    )

    assert filtered == []
