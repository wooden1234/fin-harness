import pytest

from retrieval.core.filters import reload_filter_config
from retrieval.core.kb_contract import (
    RetrievalTrace,
    SchemaGateError,
    apply_on_empty_policy,
    missing_required_fields,
    resolve_on_empty_policy,
    validate_chunk_metadata,
)


@pytest.fixture(autouse=True)
def _reload_profiles():
    reload_filter_config()
    yield
    reload_filter_config()


def test_required_fields_merge_common_and_kb_specific():
    missing = missing_required_fields(
        {
            "format": "pdf",
            "doc_id": "PDF-AR-CATL-2024",
            "title": "CATL 2024",
            "category": "annual_reports",
        },
        "annual_reports",
    )
    assert missing == ["fiscal_year", "ticker"]


def test_schema_gate_raises_for_missing_fiscal_year():
    with pytest.raises(SchemaGateError) as exc:
        validate_chunk_metadata(
            {
                "format": "pdf",
                "doc_id": "PDF-AR-CATL-2024",
                "title": "CATL 2024",
                "category": "annual_reports",
                "ticker": "300750",
            },
            "annual_reports",
            doc_id="PDF-AR-CATL-2024",
        )
    assert "fiscal_year" in exc.value.missing_fields


def test_schema_gate_passes_for_annual_report_chunk():
    validate_chunk_metadata(
        {
            "format": "pdf",
            "doc_id": "PDF-AR-CATL-2024",
            "title": "CATL 2024",
            "category": "annual_reports",
            "ticker": "300750",
            "fiscal_year": 2024,
        },
        "annual_reports",
    )


def test_resolve_on_empty_policy_from_kb_config():
    assert resolve_on_empty_policy(["annual_reports"]) == "abstain"
    assert resolve_on_empty_policy(None) == "abstain"


def test_apply_on_empty_abstain_with_reason():
    hits, trace = apply_on_empty_policy(
        [],
        query="宁德时代2026年报",
        filters={"category": "annual_reports", "year": 2026},
        categories=["annual_reports"],
        vector_hits=0,
        lexical_hits=0,
    )
    assert hits == []
    assert trace.abstained is True
    assert trace.on_empty_policy == "abstain"
    assert trace.abstain_reason == "no_hits_after_metadata_filter"


def test_apply_on_empty_keeps_hits():
    payload = [{"text": "ok"}]
    hits, trace = apply_on_empty_policy(
        payload,
        query="test",
        filters={"category": "annual_reports"},
        categories=["annual_reports"],
        vector_hits=1,
        lexical_hits=0,
    )
    assert hits == payload
    assert trace.abstained is False
    assert trace.abstain_reason is None


def test_retrieval_trace_dataclass():
    trace = RetrievalTrace(query="q", filters={}, categories=["policy"])
    assert trace.final_hits == 0
