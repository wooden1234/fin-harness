"""Agent 运行指标聚合测试。"""

from app.services.agent.agent_run_service import summarize_run_snapshots


def test_summarize_run_snapshots_exposes_quantiles_and_low_cardinality() -> None:
    summary = summarize_run_snapshots(
        [
            {
                "content": "回答一",
                "citations": [{"source": "A"}],
                "route": "answer",
                "execution_status": "completed",
                "budget_tier": "light",
                "duration_ms": 1000,
                "conflict_detected": 1,
                "conflict_resolved": 1,
            },
            {
                "content": "回答二",
                "citations": [],
                "route": "answer",
                "execution_status": "partial",
                "budget_tier": "light",
                "duration_ms": 3000,
                "completed_with_gaps": True,
            },
            {
                "content": "",
                "citations": [],
                "route": "clarify",
                "execution_status": "completed",
                "budget_tier": "standard",
                "duration_ms": 2000,
            },
        ]
    )

    assert summary["sample_size"] == 3
    assert summary["duration_ms"] == {"count": 3, "p50": 2000.0, "p95": 3000.0}
    assert summary["by_budget"]["light"]["p95"] == 3000.0
    assert summary["routes"] == {"answer": 2, "clarify": 1}
    assert summary["citation_coverage_ratio"] == 0.3333
    assert summary["answer_availability_ratio"] == 0.6667
    assert summary["conflict_detected"] == 1
    assert summary["conflict_resolved"] == 1
    assert summary["completed_with_gaps"] == 1
