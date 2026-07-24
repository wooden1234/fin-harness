"""financial_query_agent 内部服务统一出口。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.services.entity_resolver import EntityResolver
from agents.finance_agent.financial_query_agent.services.query_router import (
    FinancialQueryRouter,
    FinancialQueryTemplate,
)
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialFactQuery,
    FinancialQueryIntent,
    FinancialSqlResultRow,
    GeneratedFinancialSql,
)

__all__ = [
    "EntityResolver",
    "FinancialFactQuery",
    "FinancialFactSearchExecutor",
    "FinancialFactService",
    "FinancialCitationBuilder",
    "FinancialQueryIntent",
    "FinancialQueryRouter",
    "FinancialQueryTemplate",
    "FinancialSqlExecutor",
    "FinancialSqlResultRow",
    "FinancialSqlTemplateRegistry",
    "FinancialTemplateExecutor",
    "FinancialResultFormatter",
    "GeneratedFinancialSql",
    "SqlValidationError",
]


def __getattr__(name: str):
    if name == "FinancialCitationBuilder":
        from agents.finance_agent.financial_query_agent.services.citation_builder import (
            FinancialCitationBuilder,
        )

        return FinancialCitationBuilder
    if name == "FinancialResultFormatter":
        from agents.finance_agent.financial_query_agent.services.result_formatter import (
            FinancialResultFormatter,
        )

        return FinancialResultFormatter
    if name in {"FinancialFactSearchExecutor"}:
        from agents.finance_agent.financial_query_agent.services.fact_search_executor import (
            FinancialFactSearchExecutor,
        )

        return FinancialFactSearchExecutor
    if name == "FinancialFactService":
        from agents.finance_agent.financial_query_agent.services.fact_service import (
            FinancialFactService,
        )

        return FinancialFactService
    if name in {"FinancialSqlExecutor", "SqlValidationError"}:
        from agents.finance_agent.financial_query_agent.services.sql_executor import (
            FinancialSqlExecutor,
            SqlValidationError,
        )

        return {"FinancialSqlExecutor": FinancialSqlExecutor, "SqlValidationError": SqlValidationError}[name]
    if name == "FinancialSqlTemplateRegistry":
        from agents.finance_agent.financial_query_agent.services.sql_templates import (
            FinancialSqlTemplateRegistry,
        )

        return FinancialSqlTemplateRegistry
    if name == "FinancialTemplateExecutor":
        from agents.finance_agent.financial_query_agent.services.template_executor import (
            FinancialTemplateExecutor,
        )

        return FinancialTemplateExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
