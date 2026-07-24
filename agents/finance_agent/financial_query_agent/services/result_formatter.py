"""结构化财务事实与 SQL 行的展示格式化。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.models.finance.annual_financial_fact import AnnualFinancialFact
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow


class FinancialResultFormatter:
    """只负责把查询结果转换为面向用户的文本。"""

    @staticmethod
    def _document(fact: AnnualFinancialFact) -> Any:
        table = getattr(fact, "table", None)
        return getattr(table, "document", None) if table is not None else None

    @staticmethod
    def _table(fact: AnnualFinancialFact) -> Any:
        return getattr(fact, "table", None)

    @classmethod
    def display_company(cls, fact: AnnualFinancialFact) -> str:
        document = cls._document(fact)
        company = getattr(document, "company", None) if document is not None else None
        if company is not None and getattr(company, "name", None):
            return company.name
        title = getattr(fact, "title", None) or getattr(document, "title", "") or ""
        if " Annual Report" in title:
            return title.split(" Annual Report")[0]
        ticker = getattr(fact, "ticker", None) or getattr(company, "ticker", None)
        return title or ticker or "未知公司"

    @staticmethod
    def display_metric_name(fact: AnnualFinancialFact) -> str:
        metric = getattr(fact, "metric", None)
        return getattr(fact, "metric_name", None) or getattr(metric, "canonical_name", "未知指标")

    @classmethod
    def fact_year(cls, fact: AnnualFinancialFact) -> int | str:
        document = cls._document(fact)
        return (
            fact.period_year
            or getattr(fact, "fiscal_year", None)
            or getattr(document, "fiscal_year", None)
            or "未知年份"
        )

    @classmethod
    def fact_year_sort_key(cls, fact: AnnualFinancialFact, *, desc: bool) -> tuple[int, int]:
        year = cls.fact_year(fact)
        if isinstance(year, int):
            return (0, -year if desc else year)
        return (1, 0)

    @staticmethod
    def display_value(fact: AnnualFinancialFact) -> str:
        if fact.raw_value:
            return fact.raw_value
        if fact.value is None:
            return "—"
        if isinstance(fact.value, Decimal):
            normalized = fact.value.normalize()
            return format(normalized, "f").rstrip("0").rstrip(".")
        return str(fact.value)

    @staticmethod
    def display_sql_value(row: FinancialSqlResultRow) -> str:
        return row.raw_value or row.value or "—"

    @classmethod
    def format_answer(cls, facts: list[AnnualFinancialFact]) -> str:
        if not facts:
            return "（数据库中未找到匹配的财务指标，建议改查 PDF 文档库。）"
        lines: list[str] = []
        for fact in facts:
            document = cls._document(fact)
            table = cls._table(fact)
            page_num = getattr(fact, "page_num", None) or getattr(table, "page_num", None)
            page = f"第{page_num}页" if page_num else "未知页码"
            source = getattr(fact, "source", None) or getattr(document, "source", "")
            currency = f"，{fact.currency}" if fact.currency else ""
            lines.append(
                f"{cls.display_company(fact)} {cls.fact_year(fact)}年 "
                f"{cls.display_metric_name(fact)}为 {cls.display_value(fact)}"
                f"{fact.unit or ''}{currency}（来源：{source} {page}）"
            )
        return "\n".join(lines)

    @classmethod
    def format_sql_answer(
        cls,
        rows: list[FinancialSqlResultRow],
        *,
        include_source: bool = False,
    ) -> str:
        if not rows:
            return "（数据库中未找到匹配的财务指标，建议改查 PDF 文档库。）"
        lines: list[str] = []
        for row in rows:
            year = row.period_year or row.fiscal_year or "未知年份"
            value = cls.display_sql_value(row)
            currency = f"，{row.currency}" if row.currency else ""
            line = f"{row.company_name} {year}年 {row.metric_name}为 {value}{row.unit}{currency}"
            if include_source:
                page = f"第{row.page_num}页" if row.page_num else "未知页码"
                line += f"（来源：{row.source} {page}）"
            lines.append(line)
        return "\n".join(lines)


__all__ = ["FinancialResultFormatter"]
