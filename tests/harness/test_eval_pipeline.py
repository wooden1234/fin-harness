from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from evals.adapter import HarnessEvalAdapter
from evals.dataset import load_cases, select_cases
from evals.evaluators.evidence import evaluate_evidence
from evals.evaluators.fact import evaluate_facts, normalized_numeric
from evals.upload_dataset import upload_dataset
from harness.tools.providers import product_definitions


def _score(results: list[dict], key: str):
    return next(item["score"] for item in results if item["key"] == key)


def test_money_units_normalize_to_same_base_value() -> None:
    assert normalized_numeric(1, "亿元") == (Decimal("100000000"), "money_base")
    assert normalized_numeric(10000, "万元") == (Decimal("100000000"), "money_base")
    assert normalized_numeric(100000000, "元") == (Decimal("100000000"), "money_base")


def test_percent_and_percentage_points_are_not_interchangeable() -> None:
    reference = {
        "required_facts": [
            {"subject": "A", "period": "2024", "metric": "毛利率差", "value": 5, "unit": "百分点"}
        ]
    }
    outputs = {
        "facts": [
            {"subject": "A", "period": "2024", "metric": "毛利率差", "value": 5, "unit": "%"}
        ]
    }
    results = evaluate_facts(outputs, reference)
    assert _score(results, "fact_coverage") == 0.0


def test_preferred_document_does_not_imply_source_lock() -> None:
    reference = {
        "evidence": {"required": True, "preferred_doc_ids": ["gold-doc"]},
        "routing_contract": {"source_policy": "default"},
    }
    outputs = {
        "citations": [
            {"doc_id": "other-authoritative-doc", "node_id": "node-1", "source_type": "finance"}
        ]
    }
    results = evaluate_evidence(outputs, reference)
    assert _score(results, "evidence_coverage") == 1.0
    assert _score(results, "preferred_source_match") == 0.0
    assert not any(item["key"] == "source_lock_hard_failure" for item in results)


def test_source_lock_violation_is_hard_failure() -> None:
    reference = {
        "evidence": {"required": True},
        "routing_contract": {
            "source_policy": "source_locked",
            "allowed_source_types": ["company_filing"],
        },
    }
    outputs = {
        "citations": [
            {"url": "https://example.invalid/news", "source_type": "news"}
        ]
    }
    results = evaluate_evidence(outputs, reference)
    assert _score(results, "source_lock_hard_failure") == 0.0


def test_dangling_citation_link_is_hard_failure() -> None:
    reference = {"evidence": {"required": True}}
    outputs = {
        "citations": [{"evidence_id": "ev-1", "node_id": "node-1"}],
        "claim_evidence_links": [{"claim_id": "claim-1", "evidence_id": "invented"}],
    }
    results = evaluate_evidence(outputs, reference)
    assert _score(results, "evidence_support") == 0.0
    assert _score(results, "citation_hard_failure") == 0.0


@pytest.mark.asyncio
async def test_multiturn_reuses_one_conversation_session() -> None:
    class RecordingAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def prompt(self, text: str):
            self.calls += 1
            return SimpleNamespace(
                published_answer=f"第{self.calls}轮",
                finish_reason="completed",
                error=None,
                waiting_approval=False,
                session_id="session-one",
                run_id=f"run-{self.calls}",
                events=[],
            )

    class RecordingManager:
        def __init__(self) -> None:
            self.agent = RecordingAgent()
            self.conversation_ids: list[str] = []
            self.store = SimpleNamespace()

        async def get(self, **kwargs):
            self.conversation_ids.append(kwargs["conversation_id"])
            return self.agent

    manager = RecordingManager()
    adapter = HarnessEvalAdapter(manager=manager)
    result = await adapter.run_case(
        {
            "turns": [
                {"role": "user", "content": "先问"},
                {"role": "assistant", "expected_behavior": "ignored fixture"},
                {"role": "user", "content": "再问"},
            ]
        },
        case_id="multi",
    )
    assert len(result["turns"]) == 2
    assert result["turns"][0]["run_id"] != result["turns"][1]["run_id"]
    assert manager.conversation_ids == [result["conversation_id"]]
    assert result["session_id"] == "session-one"


def test_wrong_subject_and_period_raise_fact_hard_failure() -> None:
    reference = {
        "required_facts": [
            {"subject": "寒武纪", "period": "2024", "metric": "营业收入", "value": 100, "unit": "元"}
        ]
    }
    outputs = {
        "facts": [
            {"subject": "龙芯中科", "period": "2025", "metric": "营业收入", "value": 100, "unit": "元"}
        ]
    }
    results = evaluate_facts(outputs, reference)
    assert _score(results, "fact_coverage") == 0.0
    assert _score(results, "fact_hard_failure") == 0.0


def test_missing_required_fact_reduces_coverage() -> None:
    reference = {
        "required_facts": [
            {"subject": "寒武纪", "period": "2024", "metric": "营业收入", "value": 100, "unit": "元"},
            {"subject": "寒武纪", "period": "2024", "metric": "营业收入同比", "value": 10, "unit": "%"},
        ]
    }
    outputs = {
        "facts": [
            {"subject": "寒武纪", "period": "2024", "metric": "营业收入", "value": 100, "unit": "元"}
        ]
    }
    results = evaluate_facts(outputs, reference)
    assert _score(results, "fact_coverage") == 0.5
    assert _score(results, "fact_accuracy") == 0.0


def test_dataset_merge_and_default_calibration_selection() -> None:
    cases = load_cases()
    assert len(cases) == 50
    assert all("routing_contract" in item["reference_outputs"] for item in cases)
    selected = select_cases(cases)
    assert len(selected) == 10
    assert selected[0]["id"] == "core-001"


def test_dataset_upload_skips_existing_case_ids() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.dataset = SimpleNamespace(id="dataset-1")
            self.created: list[dict] = []

        def read_dataset(self, *, dataset_name: str):
            return self.dataset

        def list_examples(self, *, dataset_id: str):
            return [SimpleNamespace(metadata={"case_id": "core-001"})]

        def create_examples(self, *, dataset_id: str, examples: list[dict]):
            self.created.extend(examples)

    client = FakeClient()
    result = upload_dataset(client=client)
    assert result == {"dataset_name": "fin-harness-core-v1", "created": 49, "skipped": 1}
    assert len(client.created) == 49
    assert all("case_id" in item["metadata"] for item in client.created)


def test_legacy_finance_fact_tool_is_not_exposed_to_product_agent() -> None:
    tools = {item.tool_id: item for item in product_definitions()}
    assert "finance.fact.lookup" not in tools
    assert "knowledge.fact.lookup" in tools
    assert "iwencai.finance.query" in tools
