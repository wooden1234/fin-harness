"""fin_core 结构化财务数据白名单。"""

from __future__ import annotations

from dataclasses import dataclass

FIN_CORE_SCHEMA = "fin_core"

ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        f"{FIN_CORE_SCHEMA}.financial_companies",
        f"{FIN_CORE_SCHEMA}.annual_report_documents",
        f"{FIN_CORE_SCHEMA}.annual_financial_tables",
        f"{FIN_CORE_SCHEMA}.financial_metrics",
        f"{FIN_CORE_SCHEMA}.canonical_metrics",
        f"{FIN_CORE_SCHEMA}.canonical_metric_aliases",
        f"{FIN_CORE_SCHEMA}.company_metric_mappings",
        f"{FIN_CORE_SCHEMA}.raw_table_cells",
        f"{FIN_CORE_SCHEMA}.annual_financial_facts",
    }
)


@dataclass(frozen=True)
class FinCoreTableDefinition:
    schema: str
    name: str
    description: str
    columns: tuple[str, ...]
    primary_key: str
    foreign_keys: tuple[str, ...] = ()


FIN_CORE_TABLES: dict[str, FinCoreTableDefinition] = {
    "financial_companies": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="financial_companies",
        description="公司维度表",
        columns=("id", "company_key", "name", "ticker", "created_at", "updated_at"),
        primary_key="id",
    ),
    "annual_report_documents": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="annual_report_documents",
        description="年报文档元数据",
        columns=("id", "doc_id", "company_id", "title", "fiscal_year", "source", "created_at", "updated_at"),
        primary_key="id",
        foreign_keys=("company_id -> financial_companies.id",),
    ),
    "annual_financial_tables": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="annual_financial_tables",
        description="文档内财务表格分块",
        columns=(
            "id",
            "document_id",
            "chunk_index",
            "page_num",
            "section",
            "table_kind",
            "raw_table_text",
            "created_at",
            "updated_at",
        ),
        primary_key="id",
        foreign_keys=("document_id -> annual_report_documents.id",),
    ),
    "financial_metrics": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="financial_metrics",
        description="PDF 抽取出的 source metric 字典，不等同于统一业务口径",
        columns=("id", "canonical_name", "aliases", "statement_type", "created_at", "updated_at"),
        primary_key="id",
    ),
    "canonical_metrics": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="canonical_metrics",
        description="统一财务指标口径注册表",
        columns=(
            "code",
            "name",
            "statement_type",
            "value_type",
            "default_unit",
            "description",
            "is_active",
            "created_at",
            "updated_at",
        ),
        primary_key="code",
    ),
    "canonical_metric_aliases": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="canonical_metric_aliases",
        description="统一指标别名表，用于用户指标词解析",
        columns=(
            "id",
            "canonical_code",
            "alias",
            "normalized_alias",
            "source",
            "priority",
            "is_active",
            "created_at",
            "updated_at",
        ),
        primary_key="id",
        foreign_keys=("canonical_code -> canonical_metrics.code",),
    ),
    "company_metric_mappings": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="company_metric_mappings",
        description="公司级 canonical 指标到 source metric 的映射",
        columns=(
            "id",
            "company_id",
            "canonical_code",
            "source_metric_id",
            "source_metric_name",
            "statement_type",
            "valid_from_year",
            "valid_to_year",
            "priority",
            "confidence",
            "mapping_source",
            "review_status",
            "is_active",
            "created_at",
            "updated_at",
        ),
        primary_key="id",
        foreign_keys=(
            "company_id -> financial_companies.id",
            "canonical_code -> canonical_metrics.code",
            "source_metric_id -> financial_metrics.id",
        ),
    ),
    "raw_table_cells": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="raw_table_cells",
        description="PDF 表格单元格级抽取证据",
        columns=(
            "id",
            "table_id",
            "document_id",
            "page_num",
            "row_index",
            "col_index",
            "row_header",
            "col_header",
            "cell_text",
            "normalized_value",
            "unit",
            "currency",
            "bbox_json",
            "extractor",
            "extract_version",
            "confidence",
            "created_at",
            "updated_at",
        ),
        primary_key="id",
        foreign_keys=(
            "table_id -> annual_financial_tables.id",
            "document_id -> annual_report_documents.id",
        ),
    ),
    "annual_financial_facts": FinCoreTableDefinition(
        schema=FIN_CORE_SCHEMA,
        name="annual_financial_facts",
        description="窄表事实：单表单指标单期间数值；新查询应优先使用 canonical_code 和发布状态",
        columns=(
            "id",
            "table_id",
            "metric_id",
            "canonical_code",
            "source_cell_id",
            "row_index",
            "period_label",
            "period_year",
            "period_type",
            "value",
            "raw_value",
            "unit",
            "currency",
            "raw_row",
            "confidence",
            "quality_status",
            "review_status",
            "validation_errors",
            "extract_version",
            "is_published",
            "created_at",
            "updated_at",
        ),
        primary_key="id",
        foreign_keys=(
            "table_id -> annual_financial_tables.id",
            "metric_id -> financial_metrics.id",
            "canonical_code -> canonical_metrics.code",
            "source_cell_id -> raw_table_cells.id",
        ),
    ),
}

STANDARD_JOIN_SQL = f"""
FROM {FIN_CORE_SCHEMA}.annual_financial_facts AS fact
JOIN {FIN_CORE_SCHEMA}.annual_financial_tables AS table_ctx
  ON table_ctx.id = fact.table_id
JOIN {FIN_CORE_SCHEMA}.annual_report_documents AS document
  ON document.id = table_ctx.document_id
JOIN {FIN_CORE_SCHEMA}.financial_metrics AS metric
  ON metric.id = fact.metric_id
LEFT JOIN {FIN_CORE_SCHEMA}.financial_companies AS company
  ON company.id = document.company_id
LEFT JOIN {FIN_CORE_SCHEMA}.canonical_metrics AS canonical_metric
  ON canonical_metric.code = fact.canonical_code
""".strip()

BASE_FACT_FILTERS = """
  AND fact.period_label IS NOT NULL
  AND fact.period_label != ''
  AND fact.period_label NOT LIKE 'value_%'
  AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
  AND (fact.period_type IS NULL OR fact.period_type NOT IN ('change_rate', 'unknown'))
""".strip()

TEMPLATE_SELECT_SQL = f"""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  document.fiscal_year AS fiscal_year,
  fact.period_year AS period_year,
  fact.period_label AS period_label,
  fact.period_type AS period_type,
  metric.canonical_name AS metric_name,
  COALESCE(canonical_metric.name, '') AS canonical_metric_name,
  COALESCE(fact.raw_value, '') AS raw_value,
  COALESCE(CAST(fact.value AS TEXT), '') AS value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(fact.currency, '') AS currency,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
{STANDARD_JOIN_SQL}
WHERE document.company_id IN :company_ids
  AND fact.metric_id IN :metric_ids
{BASE_FACT_FILTERS}
""".strip()


def schema_prompt() -> str:
    lines = [f"Schema: {FIN_CORE_SCHEMA}", ""]
    for table in FIN_CORE_TABLES.values():
        cols = ", ".join(table.columns)
        fks = f"; FK: {', '.join(table.foreign_keys)}" if table.foreign_keys else ""
        lines.append(f"- {table.schema}.{table.name}: {table.description}; columns={cols}{fks}")
    lines.extend(["", "推荐 Join 路径：", STANDARD_JOIN_SQL, "", "模板查询过滤口径：", BASE_FACT_FILTERS])
    return "\n".join(lines)


__all__ = [
    "ALLOWED_TABLES",
    "BASE_FACT_FILTERS",
    "FIN_CORE_SCHEMA",
    "FIN_CORE_TABLES",
    "FinCoreTableDefinition",
    "STANDARD_JOIN_SQL",
    "TEMPLATE_SELECT_SQL",
    "schema_prompt",
]
