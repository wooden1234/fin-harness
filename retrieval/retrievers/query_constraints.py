"""纯规则查询约束解析器，不调用大模型。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from retrieval.core.filters import MetadataFilters, query_infer_allowed_keys

_YEAR_RE = re.compile(r"(?<![\dA-Za-z_-])(20\d{2})(?!\d)")
_TICKER_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_DOC_ID_RE = re.compile(
    r"(?<![A-Z0-9])(PDF-[A-Z0-9][A-Z0-9_-]*)(?![A-Z0-9])",
    re.IGNORECASE,
)
_REPORT_CONTEXT_RE = re.compile(r"(?:年报|年度报告|财报|财务报告|报告期|研报|研究报告)")


@dataclass(frozen=True)
class QueryConstraint:
    field: str
    value: Any
    operator: str
    role: str
    source: str
    confidence: float


@dataclass(frozen=True)
class StructuredQueryPlan:
    filters: MetadataFilters
    constraints: tuple[QueryConstraint, ...]
    soft_constraints: tuple[QueryConstraint, ...]
    unresolved: tuple[str, ...]
    reason: str


class RuleQueryConstraintParser:
    """从明确文本中提取安全的元数据约束；不猜测实体别名和年份语义。"""

    def parse(
        self,
        query: str,
        *,
        knowledge_bases: list[str] | None = None,
    ) -> StructuredQueryPlan:
        text = str(query or "").strip()
        categories = [str(value) for value in (knowledge_bases or []) if value]
        allowed = query_infer_allowed_keys(categories or None)
        filters: MetadataFilters = {}
        constraints: list[QueryConstraint] = []
        unresolved: list[str] = []

        doc_match = _DOC_ID_RE.search(text)
        if doc_match and "doc_id" in allowed:
            value = doc_match.group(1).upper()
            filters["doc_id"] = value
            constraints.append(QueryConstraint("doc_id", value, "eq", "document", "explicit", 1.0))

        ticker_match = _TICKER_RE.search(text)
        if ticker_match and "ticker" in allowed:
            value = ticker_match.group(1)
            filters["ticker"] = value
            constraints.append(QueryConstraint("ticker", value, "eq", "entity", "explicit", 1.0))

        year_match = _YEAR_RE.search(text)
        if year_match:
            year = int(year_match.group(1))
            report_context = bool(_REPORT_CONTEXT_RE.search(text))
            report_kb = any(category in {"annual_reports", "research_reports"} for category in categories)
            if "year" in allowed and (report_context or report_kb):
                filters["year"] = year
                constraints.append(
                    QueryConstraint("year", year, "eq", "reporting_year", "explicit", 0.98)
                )
            elif not report_context and not report_kb:
                unresolved.append("year_role")

        return StructuredQueryPlan(
            filters=filters,
            constraints=tuple(constraints),
            soft_constraints=(),
            unresolved=tuple(unresolved),
            reason="rule_constraints" if filters else "rule_empty",
        )


_default_parser = RuleQueryConstraintParser()


def parse_query_constraints(
    query: str,
    *,
    knowledge_bases: list[str] | None = None,
) -> StructuredQueryPlan:
    return _default_parser.parse(query, knowledge_bases=knowledge_bases)
