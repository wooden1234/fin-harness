"""Harness 确定性评测测试。"""

from harness.evaluator import evaluate_run


def test_evaluator_accepts_grounded_result_within_slo() -> None:
    report = evaluate_run(
        {
            "final_answer": "该产品满足年度消费条件时可减免年费。",
            "route": "answer",
            "execution_status": "completed",
            "expected_route": "answer",
            "citation_required": True,
            "citations": [
                {
                    "source": "产品说明",
                    "snippet": "满足年度消费条件可减免次年年费。",
                }
            ],
            "claims": [{"claim_id": "c1", "claim_type": "fact"}],
            "claim_evidence_links": [
                {"claim_id": "c1", "evidence_id": "e1"}
            ],
            "latency_ms": 4200,
            "slo_ms": 6000,
        }
    )

    assert report["passed"] is True
    assert report["metrics"]["citation_count"] == 1


def test_evaluator_rejects_empty_soft_timeout_and_missing_citation() -> None:
    report = evaluate_run(
        {
            "route": "answer",
            "execution_status": "soft_timeout",
            "citation_required": True,
            "latency_ms": 7000,
            "slo_ms": 6000,
        }
    )

    assert report["passed"] is False
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {"answer_available", "citation_provenance", "latency_slo"}
