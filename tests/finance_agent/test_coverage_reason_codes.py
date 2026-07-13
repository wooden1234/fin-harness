"""coverage reason_code 与模板映射测试。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.semantic.coverage_resolver import (
    CoverageResolver,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CompanyCoverage,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.reason_codes import (
    REASON_TEMPLATES,
    render_coverage_reasons,
)


def test_render_coverage_reasons_for_unavailable_code():
    clarify, unavailable = render_coverage_reasons("ANNUAL_DATA_NOT_FOUND")
    assert clarify == ""
    assert unavailable == REASON_TEMPLATES["ANNUAL_DATA_NOT_FOUND"]


def test_render_coverage_reasons_for_clarify_code():
    clarify, unavailable = render_coverage_reasons("GRANULARITY_CLARIFY_NEEDED")
    assert clarify == REASON_TEMPLATES["GRANULARITY_CLARIFY_NEEDED"]
    assert unavailable == ""


def test_render_metric_unrecognized_includes_requested_metric():
    clarify, unavailable = render_coverage_reasons(
        "METRIC_UNRECOGNIZED",
        requested_metric="未知指标",
    )
    assert unavailable == "无法识别指标：未知指标"
    assert clarify == ""


def test_aggregate_status_quarter_only_returns_reason_code():
    coverages = [
        CompanyCoverage(
            company_key="Tencent",
            company_id=1,
            metric_id=10,
            canonical_metric_code="REVENUE",
            metric_name="收入",
            selected_strategy="quarter_only",
        )
    ]
    status, answer_policy, reason_code = CoverageResolver._aggregate_status(
        coverages,
        query_type="lookup",
    )
    assert status == "unavailable"
    assert answer_policy == "unavailable"
    assert reason_code == "QUARTER_ONLY_NO_ANNUAL"
