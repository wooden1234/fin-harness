"""financial_query 的 SQL 模板注册与参数构建（兼容层）。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist import (
    BuiltPredefinedSql,
    PREDEFINED_SQL_DICT,
    PredefinedTemplateRegistry,
    template_catalog_text,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist.descriptions import (
    EXAMPLE_QUESTIONS,
    QUERY_DESCRIPTIONS,
    REQUIRED_FIELDS,
)


@dataclass(frozen=True)
class FinancialSqlTemplateDefinition:
    template_id: str
    description: str
    required_fields: tuple[str, ...]
    examples: tuple[str, ...]
    sql: str


@dataclass(frozen=True)
class BuiltFinancialSqlTemplate:
    template_id: str
    sql: str
    params: dict[str, object]
    missing_fields: list[str]


def _to_legacy_definition(template_id: str) -> FinancialSqlTemplateDefinition:
    definition = PREDEFINED_SQL_DICT[template_id]
    return FinancialSqlTemplateDefinition(
        template_id=template_id,
        description=QUERY_DESCRIPTIONS[template_id],
        required_fields=REQUIRED_FIELDS[template_id],
        examples=EXAMPLE_QUESTIONS[template_id],
        sql=definition.sql,
    )


def _to_legacy_built(built: BuiltPredefinedSql) -> BuiltFinancialSqlTemplate:
    return BuiltFinancialSqlTemplate(
        template_id=built.template_id,
        sql=built.sql,
        params=built.params,
        missing_fields=built.missing_fields,
    )


class FinancialSqlTemplateRegistry:
    _TEMPLATES = {template_id: _to_legacy_definition(template_id) for template_id in PREDEFINED_SQL_DICT}

    @classmethod
    def template_examples(cls) -> str:
        return template_catalog_text()

    @classmethod
    def get(cls, template_id: str) -> FinancialSqlTemplateDefinition | None:
        if template_id not in cls._TEMPLATES:
            return None
        return cls._TEMPLATES[template_id]

    @classmethod
    def valid_template_ids(cls) -> set[str]:
        return PredefinedTemplateRegistry.valid_template_ids()

    @classmethod
    async def build(
        cls,
        template_id: str,
        query: FinancialQueryIntent,
        *,
        limit: int = 5,
    ) -> BuiltFinancialSqlTemplate:
        built = await PredefinedTemplateRegistry.build(template_id, query, limit=limit)
        return _to_legacy_built(built)


__all__ = [
    "BuiltFinancialSqlTemplate",
    "FinancialSqlTemplateDefinition",
    "FinancialSqlTemplateRegistry",
]
