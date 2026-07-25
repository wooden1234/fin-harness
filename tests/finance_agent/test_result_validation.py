"""text_to_sql 结果校验单元测试。"""

from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
    GeneratedFinancialSql,
    QueryContract,
)
from agents.finance_agent.financial_query_agent.text_to_sql.validation import (
    validate_generated_sql,
    validate_query_result,
)

_FACT_SQL = """
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
"""


def test_validate_generated_sql_passes_without_canonical_path_for_result_layer_check():
    result = validate_generated_sql(
        _FACT_SQL,
        params={"metric_name": "营业收入", "limit": 5},
    )

    assert result.ok


def test_query_contract_normalizes_legacy_single_lookup_operation():
    contract = QueryContract(operation="query_single")

    assert contract.operation == "point_lookup"


def test_generated_sql_normalizes_legacy_route_metric_and_sql_fields():
    generated = GeneratedFinancialSql(
        generated_sql="SELECT 1",
        route="generate",
        query_contract={"metrics": [{"canonical_code": "REVENUE"}]},
    )

    assert generated.sql == "SELECT 1"
    assert generated.route == "execute"
    assert generated.query_contract.metrics == ["REVENUE"]


def test_generated_sql_normalizes_legacy_default_ratio_contract():
    generated = GeneratedFinancialSql(
        sql="SELECT 1",
        route="default",
        query_contract={
            "canonical_code": ["RND_EXPENSE", "REVENUE"],
            "operation": "ratio",
        },
    )

    assert generated.route == "execute"
    assert generated.query_contract.metrics == ["RND_EXPENSE", "REVENUE"]
    assert generated.query_contract.operation == "aggregate"


def test_generated_sql_normalizes_correction_aliases():
    generated = GeneratedFinancialSql(
        sql="SELECT 1",
        route="correct",
        query_contract={
            "canonical_codes": ["REVENUE"],
            "operation": "point_query",
        },
    )

    assert generated.route == "execute"
    assert generated.query_contract.metrics == ["REVENUE"]
    assert generated.query_contract.operation == "point_lookup"


def test_validate_query_result_flags_empty_point_lookup():
    result = validate_query_result(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[],
    )

    assert not result.ok
    assert result.error_type == "result_empty"
    assert "0 行" in result.error


def test_validate_query_result_allows_empty_list_question():
    result = validate_query_result(
        question="2024年有哪些公司",
        sql="SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
        rows=[],
    )

    assert result.ok


def test_validate_query_result_flags_missing_metric_values():
    result = validate_query_result(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="CATL",
                period_year=2024,
                metric_name="营业收入",
                raw_value="",
                value="",
            )
        ],
    )

    assert not result.ok
    assert result.error_type == "result_schema"
    assert "数值列" in result.error


def test_validate_query_result_accepts_complete_fact_rows():
    result = validate_query_result(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="CATL",
                period_year=2024,
                metric_name="营业收入",
                raw_value="4,000,000",
            )
        ],
    )

    assert result.ok


def test_validate_query_result_contract_accepts_company_year_metric_period():
    result = validate_query_result(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="CATL",
                period_year=2024,
                period_type="annual",
                metric_name="营业收入",
                canonical_code="REVENUE",
                raw_value="4,000,000",
            )
        ],
        contract=QueryContract(
            companies=["宁德时代"],
            years=[2024],
            metrics=["REVENUE"],
            period_type="annual",
            operation="point_lookup",
        ),
    )

    assert result.ok


def test_validate_query_result_contract_rejects_wrong_company_and_period():
    result = validate_query_result(
        question="宁德时代 2024 年营业收入是多少",
        sql=_FACT_SQL,
        rows=[
            FinancialSqlResultRow(
                company_name="Tencent",
                period_year=2024,
                period_type="quarter",
                metric_name="营业收入",
                canonical_code="REVENUE",
                raw_value="4,000,000",
            )
        ],
        contract=QueryContract(
            companies=["宁德时代"],
            years=[2024],
            metrics=["REVENUE"],
            period_type="annual",
            operation="point_lookup",
        ),
    )

    assert not result.ok
    assert result.error_type == "semantic"
