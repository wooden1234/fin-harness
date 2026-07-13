"""财务 SQL few-shot 检索器测试。"""

from agents.finance_agent.financial_query_agent.text_to_sql.retrievers.sql_examples import (
    FinancialSqlExampleRetriever,
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
