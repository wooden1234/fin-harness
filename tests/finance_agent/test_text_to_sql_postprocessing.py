"""text_to_sql 结果后处理单元测试。"""

from types import SimpleNamespace

import pytest

from agents.finance_agent.financial_query_agent.text_to_sql.execution import node as execution_node
from agents.finance_agent.financial_query_agent.services.schemas import (
    FinancialSqlResultRow,
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.execution import (
    select_best_disclosure_rows,
)
from agents.finance_agent.financial_query_agent.text_to_sql.components import nodes as nodes_module
from agents.finance_agent.financial_query_agent.text_to_sql.middleware import clarification as clarification_mod
from agents.finance_agent.financial_query_agent.text_to_sql.middleware.clarification import (
    ClarificationMiddleware,
)
from agents.finance_agent.financial_query_agent.workflows.text_to_sql import (
    text_to_sql_workflow,
)

NODES_MODULE = nodes_module
CLARIFICATION_MODULE = (
    "agents.finance_agent.financial_query_agent.text_to_sql.middleware.clarification"
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
async def test_clarification_middleware_before_generate_only_blocks_empty_question():
    result = await ClarificationMiddleware().before_generate(
        {"question": " "},
        None,
    )

    assert result is not None
    assert result.halt is True
    assert result.state_updates["missing_fields"] == ["company", "metric", "year"]


@pytest.mark.asyncio
async def test_clarification_middleware_before_generate_allows_incomplete_question():
    """缺年份等问题交给 generate 的 route=clarify，生成前不再硬拦。"""
    result = await ClarificationMiddleware().before_generate(
        {"question": "腾讯净利润是多少"},
        None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_clarification_middleware_after_generate_halts_on_clarify_route(monkeypatch):
    async def fake_build_clarification_answer(*args, **kwargs):
        return "请补充更明确的统计年份，我再继续生成查询。"

    monkeypatch.setattr(
        clarification_mod,
        "_build_clarification_answer",
        fake_build_clarification_answer,
    )

    result = await ClarificationMiddleware().after_generate(
        {"question": "腾讯净利润是多少"},
        GeneratedFinancialSql(
            sql="",
            params={},
            route="clarify",
            reason="缺少统计年份",
            missing_fields=["year"],
        ),
        None,
    )

    assert result is not None
    assert result.halt is True
    assert result.state_updates["missing_fields"] == ["year"]
    assert result.state_updates["route_reason"] == "缺少统计年份"

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
async def test_text_to_sql_workflow_clarifies_after_generate(monkeypatch):
    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="",
            params={},
            route="clarify",
            reason="缺少统计年份",
            missing_fields=["year"],
        )

    async def fake_build_clarification_answer(*args, **kwargs):
        return "请补充更明确的统计年份，我再继续生成查询。"

    monkeypatch.setattr(
        NODES_MODULE,
        "generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        f"{CLARIFICATION_MODULE}._build_clarification_answer",
        fake_build_clarification_answer,
    )

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
        NODES_MODULE,
        "generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        NODES_MODULE,
        "execute_generated_sql",
        fake_execute_generated_sql,
    )
    monkeypatch.setattr(
        NODES_MODULE,
        "format_sql_rows",
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
        NODES_MODULE,
        "generate_sql",
        fake_generate_sql,
    )
    monkeypatch.setattr(
        NODES_MODULE,
        "correct_sql",
        fake_correct_sql,
    )

    result = await text_to_sql_workflow(_workflow_state("2024年营收排名前十的公司"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["financial_query_sql_attempts"] == 2
    assert result["financial_query_validation_error"]


@pytest.mark.asyncio
async def test_text_to_sql_workflow_db_verify_retries_then_errors(monkeypatch):
    """真库失败先走 correct_sql 纠错环，用尽次数后才 execution_error。"""
    call_counts = {"execute": 0, "correct": 0}

    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql="SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
            params={"limit": 5},
            route="execute",
            reason="ok",
            missing_fields=[],
        )

    async def fake_execute_generated_sql(*args, **kwargs):
        call_counts["execute"] += 1
        raise RuntimeError("db unavailable")

    async def fake_correct_sql(*args, **kwargs):
        call_counts["correct"] += 1
        return GeneratedFinancialSql(
            sql="SELECT company.name AS company_name FROM fin_core.financial_companies AS company LIMIT :limit",
            params={"limit": 5},
            route="execute",
            reason="runtime fix failed",
            missing_fields=[],
        )

    monkeypatch.setattr(NODES_MODULE, "generate_sql", fake_generate_sql)
    monkeypatch.setattr(NODES_MODULE, "execute_generated_sql", fake_execute_generated_sql)
    monkeypatch.setattr(NODES_MODULE, "correct_sql", fake_correct_sql)

    result = await text_to_sql_workflow(_workflow_state("2024年有哪些公司"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["financial_query_sql_attempts"] == 2
    assert call_counts["execute"] == 2
    assert call_counts["correct"] == 1
    assert result["messages"][0].content


@pytest.mark.asyncio
async def test_text_to_sql_workflow_retries_on_empty_point_lookup(monkeypatch):
    """点查空结果走 validate_result → correct_sql 纠错环。"""
    call_counts = {"execute": 0, "correct": 0}

    fact_sql = """
SELECT company.name AS company_name, fact.raw_value AS raw_value
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.financial_companies AS company ON company.id = 1
LIMIT :limit
"""

    async def fake_generate_sql(*args, **kwargs):
        return GeneratedFinancialSql(
            sql=fact_sql,
            params={"limit": 5},
            route="execute",
            reason="ok",
            missing_fields=[],
        )

    async def fake_execute_generated_sql(*args, **kwargs):
        call_counts["execute"] += 1
        return []

    async def fake_correct_sql(*args, **kwargs):
        call_counts["correct"] += 1
        return GeneratedFinancialSql(
            sql=fact_sql,
            params={"limit": 5},
            route="execute",
            reason="still empty",
            missing_fields=[],
        )

    monkeypatch.setattr(NODES_MODULE, "generate_sql", fake_generate_sql)
    monkeypatch.setattr(NODES_MODULE, "execute_generated_sql", fake_execute_generated_sql)
    monkeypatch.setattr(NODES_MODULE, "correct_sql", fake_correct_sql)

    result = await text_to_sql_workflow(_workflow_state("宁德时代 2024 年营业收入是多少"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["financial_query_sql_attempts"] == 2
    assert call_counts["execute"] == 2
    assert call_counts["correct"] == 1
    assert result["messages"][0].content


@pytest.mark.asyncio
async def test_text_to_sql_workflow_maps_recursion_error_to_execution_error(monkeypatch):
    from langgraph.errors import GraphRecursionError

    async def _raise_recursion(*args, **kwargs):
        raise GraphRecursionError("recursion limit reached")

    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.workflows.text_to_sql._get_compiled_text_to_sql_graph",
        lambda: SimpleNamespace(ainvoke=_raise_recursion),
    )

    result = await text_to_sql_workflow(_workflow_state("2024年有哪些公司"))

    assert result["financial_query_next_action_sql"] == "end"
    assert result["task_results"][0]["coverage"] == "uncovered"
    assert result["task_results"][0]["fallback_reason"] == "financial_query_text_to_sql_failed"
    assert "当前问题超出安全模板" in result["messages"][0].content
