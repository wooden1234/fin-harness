"""text_to_sql SQL 校验单元测试。"""

from app.agents.finance_agent.financial_query_agent.text_to_sql.validation import (
    validate_generated_sql,
)
from app.agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)


def test_generated_financial_sql_normalizes_query_route_to_execute():
    generated = GeneratedFinancialSql.model_validate({"route": "query"})

    assert generated.route == "execute"


def test_validate_generated_sql_appends_system_limit_when_missing():
    result = validate_generated_sql(
        "SELECT fact.id FROM fin_core.annual_financial_facts AS fact WHERE fact.id = :fact_id",
        params={"fact_id": 1},
    )

    assert result.ok
    assert "LIMIT :__system_limit" in result.validated_sql


def test_validate_generated_sql_accepts_expanding_list_parameter():
    result = validate_generated_sql(
        "SELECT fact.id FROM fin_core.annual_financial_facts AS fact WHERE fact.id IN :fact_ids LIMIT :limit",
        params={"fact_ids": [1, 2], "limit": 5},
    )

    assert result.ok
    assert result.error == ""


def test_validate_generated_sql_rejects_missing_parameter():
    result = validate_generated_sql(
        "SELECT fact.id FROM fin_core.annual_financial_facts AS fact WHERE fact.id = :fact_id LIMIT :limit",
        params={"limit": 5},
    )

    assert not result.ok
    assert result.error_type == "parameter"
    assert "缺少绑定参数" in result.error
    assert "fact_id" in result.error


def test_validate_generated_sql_rejects_unused_parameter():
    result = validate_generated_sql(
        "SELECT fact.id FROM fin_core.annual_financial_facts AS fact LIMIT :limit",
        params={"limit": 5, "company_name": "CATL"},
    )

    assert not result.ok
    assert result.error_type == "parameter"
    assert "未使用参数" in result.error
    assert "company_name" in result.error


def test_validate_generated_sql_rejects_non_whitelisted_table():
    result = validate_generated_sql(
        "SELECT * FROM public.users WHERE id = :id LIMIT :limit",
        params={"id": 1, "limit": 5},
    )

    assert not result.ok
    assert result.error_type == "schema"
    assert "非白名单表" in result.error


def test_validate_generated_sql_rejects_fact_value_query_without_canonical_path():
    result = validate_generated_sql(
        """
SELECT
  company.name AS company_name,
  fact.period_year AS period_year,
  metric.canonical_name AS metric_name,
  fact.raw_value AS raw_value
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE metric.canonical_name = :metric_name
LIMIT :limit
""",
        params={"metric_name": "营业收入", "limit": 5},
    )

    assert not result.ok
    assert result.error_type == "semantic"
    assert "财务事实查数必须使用" in result.error


def test_validate_generated_sql_accepts_company_metric_mapping_canonical_path():
    result = validate_generated_sql(
        """
SELECT
  company.name AS company_name,
  fact.period_year AS period_year,
  canonical_metric.name AS metric_name,
  fact.raw_value AS raw_value
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
LIMIT :limit
""",
        params={"canonical_code": "REVENUE", "limit": 5},
    )

    assert result.ok


def test_validate_generated_sql_allows_metadata_query_without_canonical_path():
    result = validate_generated_sql(
        "SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
        params={"limit": 5},
    )

    assert result.ok
