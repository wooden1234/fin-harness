"""受限财务查询工具。

``fetch_financial_fact`` 供 knowledge 补充调用；本模块仍注册
``finance.fact.lookup`` / ``finance.query_advanced`` 以兼容旧链路，
但 Main DeepAgent 默认不再把它们作为一等公民暴露。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agents.finance_agent.financial_query_agent import _run_financial_query_agent
from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.services.fact_service import (
    FinancialFactService,
)
from agents.finance_agent.financial_query_agent.services.result_formatter import (
    FinancialResultFormatter,
)
from tools.core.base import ToolSpec
from tools.core.registry import register_tool

FactOperation = Literal["lookup", "latest", "compare", "compare_year", "trend"]


def _fact_payload(fact) -> dict[str, Any]:
    """只暴露查询所需字段，避免序列化 SQLAlchemy relationship。"""
    return {
        "company": FinancialResultFormatter.display_company(fact),
        "period_year": FinancialResultFormatter.fact_year(fact),
        "metric": FinancialResultFormatter.display_metric_name(fact),
        "value": str(getattr(fact, "raw_value", None) or getattr(fact, "value", "")),
        "unit": str(getattr(fact, "unit", "") or ""),
        "currency": str(getattr(fact, "currency", "") or ""),
    }


async def fetch_financial_fact(
    question: str,
    companies: list[str],
    metrics: list[str],
    years: list[int] | None = None,
    operation: FactOperation = "latest",
    top_k: int = 5,
) -> dict[str, Any]:
    """查询本地已入库财务事实表；未命中返回明确 error，不假装全能。"""
    intent = FinancialQueryIntent(
        companies=companies,
        metrics=metrics,
        years=years or [],
        operation=operation,
        time_scope=(
            "latest"
            if operation == "latest"
            else "single"
            if len(years or []) == 1
            else "range"
            if years
            else "unspecified"
        ),
        top_k=min(max(1, top_k), 20),
    )
    facts, route = await FinancialFactService.execute_query(question, intent)
    if route == FinancialFactService.NEEDS_CLARIFICATION_ROUTE:
        return {
            "ok": False,
            "error": "fact_query_requires_clarification",
            "route": route,
            "message": "结构化查询缺少明确公司、指标或期间，不能安全执行。",
        }
    if route == FinancialFactService.TEXT_TO_SQL_FALLBACK_ROUTE:
        return {
            "ok": False,
            "error": "fact_query_scope_unsupported",
            "route": "unsupported_structured_query",
            "message": (
                "请求超出本地事实表白名单查询形状；该入口不会生成 SQL，"
                "请改走其它来源。"
            ),
        }
    if not facts:
        return {
            "ok": False,
            "error": "fact_not_in_local_store",
            "route": route,
            "message": "本地事实表无匹配结果；请改走 PDF/问财/其它来源。",
        }
    return {
        "ok": True,
        "route": route,
        "answer": FinancialFactService.format_answer(facts),
        "facts": [_fact_payload(item) for item in facts],
        "citations": FinancialFactService.to_citations(facts),
    }


@tool(parse_docstring=True)
async def lookup_financial_fact(
    question: str,
    companies: list[str],
    metrics: list[str],
    years: list[int] | None = None,
    operation: FactOperation = "latest",
    top_k: int = 5,
) -> dict:
    """通过白名单模板读取结构化财务事实，不调用财务 Planner。

    Args:
        question: 用户财务问题。
        companies: 公司名称或代码。
        metrics: 标准财务指标名称。
        years: 财年列表，查询最新值时可为空。
        operation: 查询操作。
        top_k: 最大返回条数，最多 20。
    """
    return await fetch_financial_fact(
        question,
        companies,
        metrics,
        years=years,
        operation=operation,
        top_k=top_k,
    )


@tool(parse_docstring=True)
async def query_finance_advanced(question: str) -> dict:
    """执行一次受限高级财务查询；调用方必须另行施加 12 秒超时。

    Args:
        question: 需要聚合、筛选或比较的财务问题。
    """
    result = await _run_financial_query_agent(
        {
            "messages": [HumanMessage(content=question)],
            "financial_query_text": question,
        }
    )
    task_results = list(result.get("task_results") or [])
    covered = [
        item
        for item in task_results
        if str(item.get("coverage") or "") == "covered"
    ]
    messages = list(result.get("messages") or [])
    answer = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if content:
            answer = content if isinstance(content, str) else str(content)
            break
    return {
        "ok": bool(covered),
        "answer": answer,
        "citations": list(result.get("citations") or []),
        "task_results": covered,
        "error": "" if covered else str(
            result.get("financial_query_failure_code") or "financial_query_uncovered"
        ),
    }


register_tool(
    ToolSpec(
        tool_id="finance.fact.lookup",
        name="lookup_financial_fact",
        description="兼容入口：本地已入库财务事实窄查（Main 请用 knowledge.fact.lookup）",
        read_only=True,
        timeout_seconds=5.0,
    ),
    langchain_tool=lookup_financial_fact,
)
register_tool(
    ToolSpec(
        tool_id="finance.query_advanced",
        name="query_finance_advanced",
        description="执行一次受限高级财务聚合查询，仅供内部/旧链路使用",
        read_only=True,
        timeout_seconds=12.0,
    ),
    langchain_tool=query_finance_advanced,
)


__all__ = [
    "FactOperation",
    "fetch_financial_fact",
    "lookup_financial_fact",
    "query_finance_advanced",
]
