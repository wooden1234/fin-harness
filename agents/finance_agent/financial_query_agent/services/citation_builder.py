"""结构化财务事实的 citation 构建。"""

from __future__ import annotations

from typing import Any

from app.models.finance.annual_financial_fact import AnnualFinancialFact
from app.shared import Citation
from agents.finance_agent.financial_query_agent.services.result_formatter import (
    FinancialResultFormatter,
)
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow


class FinancialCitationBuilder:
    """只负责把查询结果映射为统一引用结构。"""

    @staticmethod
    def to_citations(facts: list[AnnualFinancialFact]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for fact in facts:
            document = FinancialResultFormatter._document(fact)
            table = FinancialResultFormatter._table(fact)
            metric_name = FinancialResultFormatter.display_metric_name(fact)
            snippet = f"{metric_name}: {fact.raw_value or fact.value}"
            if fact.unit:
                snippet = f"{snippet}{fact.unit}"
            citation: dict[str, Any] = {
                "source": (
                    getattr(fact, "source", None)
                    or getattr(document, "source", None)
                    or getattr(fact, "doc_id", None)
                    or getattr(document, "doc_id", "")
                    or ""
                ),
                "snippet": snippet[:200],
            }
            page_num = getattr(fact, "page_num", None) or getattr(table, "page_num", None)
            if page_num is not None:
                citation["page"] = page_num
            citations.append(citation)
        return citations

    @staticmethod
    def sql_rows_to_citations(
        rows: list[FinancialSqlResultRow],
        *,
        sub_task_id: str = "",
    ) -> list[Citation]:
        citations: list[Citation] = []
        for row in rows:
            source = row.source or row.doc_id
            if not source:
                continue
            snippet = f"{row.metric_name}: {row.raw_value or row.value}"
            if row.unit:
                snippet = f"{snippet}{row.unit}"
            citation: Citation = {
                "source": source,
                "snippet": snippet[:200],
                "source_type": "pdf",
                "sub_task_id": sub_task_id,
            }
            for field in ("page_num", "section", "doc_id", "document_id", "table_id", "source_cell_id"):
                value = getattr(row, field)
                if value is not None and value != "":
                    citation["page" if field == "page_num" else field] = value
            citations.append(citation)
        return citations


__all__ = ["FinancialCitationBuilder"]
