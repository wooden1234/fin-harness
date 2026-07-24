"""financial_query_agent 内部规划节点。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    database_failure_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.planner.models import (
    FinancialQueryPlan,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.normalizer import (
    build_predefined_query_intent,
)
from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection import (
    select_predefined_tool,
)
from agents.finance_agent.financial_query_agent.services.errors import FailureInfo
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

_COMPLEX_QUERY_KEYWORDS = (
    "排名",
    "前十",
    "前10",
    "最高",
    "最低",
    "占比",
    "比例",
    "份额",
    "同比",
    "环比",
    "增速",
    "增长率",
    "cagr",
    "CAGR",
    "大于",
    "小于",
    "不少于",
    "不低于",
    "不超过",
    "介于",
    "筛选",
    "过滤",
    "平均",
    "合计",
    "总和",
    "聚合",
    "排序",
)


def _looks_like_complex_query(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return True
    return any(keyword.lower() in normalized for keyword in _COMPLEX_QUERY_KEYWORDS)


def _fallback_plan(question: str) -> FinancialQueryPlan:
    """LLM 不可用时的保守兜底：默认走 text_to_sql。"""
    if _looks_like_complex_query(question):
        return FinancialQueryPlan(
            route="text_to_sql",
            reason="问题包含复杂查询特征，统一交给 text_to_sql_workflow。",
            confidence=0.9,
        )
    return FinancialQueryPlan(
        route="text_to_sql",
        reason="无法确定白名单命中，保守交给 text_to_sql_workflow。",
        confidence=0.75,
    )


async def _plan_query(
    question: str,
    config: RunnableConfig = None,
) -> tuple[
    FinancialQueryPlan,
    str | None,
    FinancialQueryIntent | None,
    FailureInfo | None,
]:
    fallback = _fallback_plan(question)
    if _looks_like_complex_query(question):
        return fallback, None, None, None

    selection = await select_predefined_tool(question, config)
    if not selection.success or selection.slots is None:
        failure = FailureInfo(
            category=selection.failure_category or "llm_parse_error",
            code=selection.failure_code or selection.error or "tool_selection_failed",
            retryable=selection.failure_retryable,
        )
        return (
            FinancialQueryPlan(
                route="text_to_sql",
                reason=selection.error or "问题未可靠命中白名单模板。",
                confidence=0.8,
            ),
            None,
            None,
            failure,
        )

    intent = build_predefined_query_intent(selection.slots)
    return (
        FinancialQueryPlan(
            route="predefined",
            reason=f"已选择白名单模板 {selection.template_id} 并完成参数抽取。",
            confidence=0.95,
        ),
        selection.template_id,
        intent,
        None,
    )


async def financial_query_planner(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """复杂问题规则直达 SQL；简单问题一次完成模板选择和字段抽取。"""
    try:
        question = str(state.get("financial_query_text") or query_from_state(state)).strip()
    except ValueError:
        question = ""
    if not question:
        logger.error("financial_query_planner missing question")
        return database_failure_output(state, step="financial_query_planner_error")

    plan, template_id, intent, failure = await _plan_query(question, config)
    logger.info(
        "financial_query_planner route={} template_id={} reason={}",
        plan.route,
        template_id,
        plan.reason,
    )
    updates = {
        "financial_query_text": question,
        "financial_query_plan_route": plan.route,
        "financial_query_plan_reason": plan.reason,
        "steps": ["financial_query_planner"],
    }
    if failure is not None:
        updates.update(
            {
                "financial_query_failure_category": failure.category,
                "financial_query_failure_code": failure.code,
                "financial_query_failure_retryable": failure.retryable,
            }
        )
    if template_id and intent is not None:
        updates["financial_query_template_id"] = template_id
        updates["financial_query_intent"] = intent
    return updates


__all__ = ["financial_query_planner"]
