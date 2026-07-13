"""text_to_sql 结果后处理单元测试。"""

from types import SimpleNamespace

import pytest

from app.agents.finance_agent.financial_query_agent.text_to_sql.execution import node as execution_node
from app.agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
    GeneratedFinancialSql,
)
from app.agents.finance_agent.financial_query_agent.text_to_sql.execution import (
    select_best_disclosure_rows,
)
from app.agents.finance_agent.financial_query_agent.text_to_sql.middleware.clarification import (
    ClarificationMiddleware,
)
from app.agents.finance_agent.financial_query_agent.workflows.text_to_sql import (
    text_to_sql_workflow,
)


def _workflow_state(question: str) -> dict:
    return {
        "messages": [],
        "sub_question": question,
        "sub_task_id": "task-1",
        "financial_query_text": question,
    }


def test_select_best_disclosure_rows_prefers_same_fiscal_year_full_year_value():
    rows = [
        FinancialSqlResultRow(
            company_id=3,
            company_name="Tencent",
            fiscal_year=2025,
            period_year=2024,
            period_label="二零二四年人民幣百萬元",
            period_type="annual",
            metric_name="收入",
            raw_value="660,257",
            page_num=4,
        ),
        FinancialSqlResultRow(
            company_id=3,
            company_name="Tencent",
            fiscal_year=2024,
            period_year=2024,
            period_label="二零二四年九月三十日",
            period_type="annual",
            metric_name="收入",
            raw_value="167,193",
            page_num=16,
        ),
        FinancialSqlResultRow(
            company_id=3,
            company_name="Tencent",
            fiscal_year=2024,
            period_year=2024,
            period_label="二零二四年",
            period_type="annual",
            metric_name="收入",
            raw_value="660,257",
            page_num=8,
        ),
    ]

    selected = select_best_disclosure_rows(rows)

    assert len(selected) == 1
    assert selected[0].fiscal_year == 2024
    assert selected[0].raw_value == "660,257"
    assert selected[0].period_label == "二零二四年"


@pytest.mark.asyncio
async def test_clarification_middleware_detects_missing_year():
    result = await ClarificationMiddleware().before_generate(
        {"question": "腾讯净利润是多少"},
        None,
    )

    assert result is not None
    assert result.halt is True
    assert result.state_updates["missing_fields"] == ["year"]


@pytest.mark.asyncio
async def test_clarification_middleware_allows_latest_question():
    result = await ClarificationMiddleware().before_generate(
        {"question": "腾讯最新净利润是多少"},
        None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_clarification_middleware_detects_missing_metric_and_scope():
    result = await ClarificationMiddleware().before_generate(
        {"question": "宁德时代 2024 年利润是多少"},
        None,
    )

    assert result is not None
    assert result.halt is True
    assert result.state_updates["missing_fields"] == ["metric", "scope"]


@pytest.mark.asyncio
async def test_execute_generated_sql_normalizes_company_name_params(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_resolve(values):
        mapping = {
            "腾讯": SimpleNamespace(name="Tencent", db_company_key="TCEHY", ticker="TCEHY"),
            "宁德时代": SimpleNamespace(name="CATL", db_company_key="CATL", ticker="CATL"),
        }
        return [mapping[value] for value in values if value in mapping]

    async def fake_run_generated_sql(sql, *, params=None, limit=5):
        captured["sql"] = sql
        captured["params"] = params
        captured["limit"] = limit
        return []

    monkeypatch.setattr(execution_node.CompanyResolver, "resolve", fake_resolve)
    monkeypatch.setattr(
        execution_node.FinancialFactService,
        "run_generated_sql",
        fake_run_generated_sql,
    )

    await execution_node.execute_generated_sql(
        "SELECT company.name AS company_name FROM fin_core.financial_companies AS company WHERE company.name IN :company_names LIMIT :limit",
        params={"company_names": ["宁德时代", "腾讯"], "limit": 5},
    )

    assert captured["params"] == {"company_names": ["CATL", "Tencent"], "limit": 5}


@pytest.mark.asyncio
async def test_text_to_sql_workflow_clarifies_before_generate():
    result = await text_to_sql_workflow(_workflow_state("腾讯净利润是多少"))

    assert result["financial_query_next_action_sql"] == "clarify"
    assert result["financial_query_missing_fields"] == ["year"]
    assert result["steps"] == ["text_to_sql"]


@pytest.mark.asyncio
async def test_text_to_sql_workflow_success_path(monkeypatch):
    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
            params={"limit": 5},
            route="execute",
            reason="ok",
            missing_fields=[],
        )

    async def fake_execute_generated_sql(*args, **kwargs):
        return [
            FinancialSqlResultRow(
                company_name="Tencent",
                metric_name="营业收入",
                raw_value="660,257",
            )
        ]

    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.execute_generated_sql",
        fake_execute_generated_sql,
    )
    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.format_sql_rows",
        lambda rows: "结构化查询成功",
    )

    result = await text_to_sql_workflow(_workflow_state("2024年有哪些公司"))

    assert result["financial_query_next_action_sql"] == "execute"
    assert result["financial_query_sql_attempts"] == 1
    assert "结构化查询成功" in result["messages"][0].content


@pytest.mark.asyncio
async def test_text_to_sql_workflow_exhausts_validation_retries(monkeypatch):
    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="DELETE FROM fin_core.financial_companies WHERE id = :id",
            params={"id": 1},
            route="execute",
            reason="unsafe",
            missing_fields=[],
        )

    async def fake_correct_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="DELETE FROM fin_core.financial_companies WHERE id = :id",
            params={"id": 1},
            route="execute",
            reason="still unsafe",
            missing_fields=[],
        )

    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.correct_sql",
        fake_correct_sql,
    )

    result = await text_to_sql_workflow(_workflow_state("2024年营收排名前十的公司"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["financial_query_sql_attempts"] == 3
    assert result["financial_query_validation_error"]


@pytest.mark.asyncio
async def test_text_to_sql_workflow_execution_error_path(monkeypatch):
    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
            params={"limit": 5},
            route="execute",
            reason="ok",
            missing_fields=[],
        )

    async def fake_execute_generated_sql(*args, **kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        "app.agents.finance_agent.financial_query_agent.text_to_sql.components.nodes.execute_generated_sql",
        fake_execute_generated_sql,
    )

    result = await text_to_sql_workflow(_workflow_state("2024年有哪些公司"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["financial_query_sql_attempts"] == 1
    assert result["messages"][0].content
