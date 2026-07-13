"""predefined 模板查询白名单。"""

from .descriptions import (
    APPROVED_CANONICAL_SCOPE_TEXT,
    COMPARE_METRIC_LOOKUP,
    COMPARE_YEAR_METRIC_LOOKUP,
    EXACT_METRIC_LOOKUP,
    LATEST_METRIC_LOOKUP,
    TREND_METRIC_LOOKUP,
    VALID_TEMPLATE_IDS,
    collect_slot_missing_fields,
    template_catalog_text,
)
from .registry import (
    BuiltPredefinedSql,
    PredefinedTemplateRegistry,
    ResolvedPredefinedQuery,
)
from .schema import ALLOWED_TABLES, FIN_CORE_SCHEMA, schema_prompt
from .sql_dict import PREDEFINED_SQL_DICT, PredefinedSqlDefinition

__all__ = [
    "ALLOWED_TABLES",
    "BuiltPredefinedSql",
    "APPROVED_CANONICAL_SCOPE_TEXT",
    "COMPARE_METRIC_LOOKUP",
    "COMPARE_YEAR_METRIC_LOOKUP",
    "EXACT_METRIC_LOOKUP",
    "FIN_CORE_SCHEMA",
    "LATEST_METRIC_LOOKUP",
    "PREDEFINED_SQL_DICT",
    "PredefinedSqlDefinition",
    "PredefinedTemplateRegistry",
    "ResolvedPredefinedQuery",
    "TREND_METRIC_LOOKUP",
    "VALID_TEMPLATE_IDS",
    "collect_slot_missing_fields",
    "schema_prompt",
    "template_catalog_text",
]
