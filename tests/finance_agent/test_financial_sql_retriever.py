"""财务 SQL few-shot 检索器测试。"""

from agents.finance_agent.financial_query_agent.text_to_sql.retrievers.sql_examples import (
    FinancialSqlExampleRetriever,
)
from agents.finance_agent.financial_query_agent.text_to_sql.generation.context import (
    build_fewshot_examples,
    build_schema_prompt,
)
from agents.finance_agent.financial_query_agent.text_to_sql.generation.node import (
    build_text_to_sql_prompt,
)


def test_retriever_prefers_quarter_examples_for_quarter_question():
    retriever = FinancialSqlExampleRetriever()
    examples = retriever.get_examples("宁德时代 2024 年一季度营业收入是多少", k=3)

    assert examples
    assert any(example.category == "季度查询" for example in examples)
    assert "period_type = 'quarter'" in examples[0].sql


def test_retriever_prefers_quarter_examples_for_q3_question():
    retriever = FinancialSqlExampleRetriever()
    examples = retriever.get_examples("腾讯 2023 年 Q3 营业收入", k=2)

    assert examples
    assert examples[0].category == "季度查询"


def test_retriever_prefers_quarter_examples_for_all_quarters_question():
    retriever = FinancialSqlExampleRetriever()
    examples = retriever.get_examples("比亚迪 2024 年各季度净利润", k=2)

    assert examples
    assert any(example.category == "季度查询" for example in examples)
    assert any("period_label" in example.sql for example in examples)


def test_retriever_keeps_annual_example_for_annual_question():
    retriever = FinancialSqlExampleRetriever()
    examples = retriever.get_examples("宁德时代 2024 年营业收入是多少", k=2)

    assert examples
    assert examples[0].category == "单指标查数"
    assert "period_type = 'annual'" in examples[0].sql or "period_type IS NULL" in examples[0].sql


def test_build_fewshot_examples_uses_one_example_by_default():
    prompt = build_fewshot_examples("宁德时代 2024 年一季度营业收入是多少")

    assert "示例1（季度查询）" in prompt
    assert "示例2（" not in prompt
    for alias in ("document_id", "table_id", "source_cell_id", "section"):
        assert f"AS {alias}" in prompt


def test_dynamic_schema_keeps_core_provenance_and_excludes_raw_cells_by_default():
    schema = build_schema_prompt("腾讯 2024 年营业收入是多少")

    for table in (
        "financial_companies",
        "annual_report_documents",
        "annual_financial_tables",
        "annual_financial_facts",
        "canonical_metrics",
    ):
        assert table in schema
    for alias in ("doc_id", "page_num", "source_cell_id", "section"):
        assert alias in schema
    assert "raw_table_cells" not in schema
    assert "company_metric_mappings" not in schema


def test_dynamic_schema_adds_only_relevant_fragments():
    quarter_schema = build_schema_prompt("宁德时代 2024 年各季度营业收入趋势")
    mapping_schema = build_schema_prompt("2024 年营业收入排名前十的公司")
    raw_schema = build_schema_prompt("应收账款坏账准备原始表格单元格")

    assert "period_type = 'quarter'" in quarter_schema
    assert "company_metric_mappings" in mapping_schema
    assert "raw_table_cells" in raw_schema


def test_schema_without_question_keeps_full_legacy_catalog():
    schema = build_schema_prompt()

    assert "company_metric_mappings" in schema
    assert "raw_table_cells" in schema
    assert "period_type = 'quarter'" in schema


def test_dynamic_text_to_sql_prompt_stays_within_budget():
    question = "宁德时代 2024 年各季度营业收入趋势"
    prompt = build_text_to_sql_prompt(
        schema_prompt=build_schema_prompt(question),
        fewshot_examples=build_fewshot_examples(question),
    )

    assert len(prompt) <= 5000
