"""确定性 market.compute 测试。"""

from __future__ import annotations

import pytest

from agents.market_compute.compute import (
    MarketComputationError,
    compute_candidate_set,
)
from agents.orchestrator.contracts import (
    CandidateSet,
    MarketFilter,
    MarketQueryPlan,
    MarketSort,
)


def _candidate_set() -> CandidateSet:
    return CandidateSet(
        dataset_id="source-1",
        universe="A股",
        provider="iwencai",
        as_of="2026-07-27",
        rows=[
            {"code": "AAA", "pe_ttm": 20, "growth": 15},
            {"code": "BBB", "pe_ttm": 35, "growth": 30},
            {"code": "CCC", "pe_ttm": 10, "growth": 8},
        ],
        evidence_ids=["e-1"],
    )


def test_compute_filters_sorts_and_limits() -> None:
    result = compute_candidate_set(
        _candidate_set(),
        MarketQueryPlan(
            universe="A股",
            filters=[MarketFilter(field="pe_ttm", operator="lt", value=30)],
            sort=[MarketSort(field="growth", direction="desc")],
            limit=1,
        ),
    )

    assert [row["code"] for row in result.rows] == ["AAA"]
    assert result.metadata["source_row_count"] == 3
    assert result.metadata["result_row_count"] == 1


def test_compute_rejects_unknown_field() -> None:
    with pytest.raises(
        MarketComputationError,
        match="unknown_market_field:missing_metric",
    ):
        compute_candidate_set(
            _candidate_set(),
            MarketQueryPlan(
                universe="A股",
                filters=[
                    MarketFilter(
                        field="missing_metric",
                        operator="gt",
                        value=0,
                    )
                ],
            ),
        )


def test_compute_rejects_enrichment_without_source_tool() -> None:
    with pytest.raises(
        MarketComputationError,
        match="enrichments_require_provider_tools",
    ):
        compute_candidate_set(
            _candidate_set(),
            MarketQueryPlan(universe="A股", enrichments=["latest_price"]),
        )
