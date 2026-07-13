"""predefined 答案格式化与口径说明。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.predefined.semantic.models import (
    CoverageResolution,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.reason_codes import (
    render_coverage_reasons,
)
from agents.finance_agent.financial_query_agent.predefined.semantic.registry_seed import (
    CANONICAL_METRICS,
)
from agents.finance_agent.financial_query_agent.services.fact_service import (
    FinancialFactService,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
)


def build_coverage_notes(coverage: CoverageResolution | None) -> str:
    if coverage is None:
        return ""
    notes: list[str] = []
    for item in coverage.company_coverages:
        canonical = CANONICAL_METRICS.get(item.canonical_metric_code)
        canonical_name = canonical.name if canonical else item.canonical_metric_code
        if item.metric_name and item.metric_name != canonical_name:
            company_label = item.company_key or "该公司"
            notes.append(f"{company_label}按「{item.metric_name}」字段映射为{canonical_name}语义")
    if coverage.answer_policy == "compare_with_mixed_source_metrics":
        notes.append("对比结果包含不同口径来源，请在解读时注意字段差异")
    elif coverage.answer_policy == "partial_compare":
        partial_note = coverage.clarify_reason or render_coverage_reasons(
            coverage.reason_code,
        )[0]
        notes.append(partial_note or "部分公司缺少可比年报数据")
    if not notes:
        return ""
    return "口径说明：\n" + "\n".join(f"- {note}" for note in notes)


def format_predefined_answer(
    rows: list[FinancialSqlResultRow],
    coverage: CoverageResolution | None = None,
) -> str:
    base = FinancialFactService.format_sql_answer(rows)
    notes = build_coverage_notes(coverage)
    if notes:
        return f"{base}\n\n{notes}"
    return base


__all__ = ["build_coverage_notes", "format_predefined_answer"]
