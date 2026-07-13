"""financial_query_agent 内部规划节点。"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import (
    database_failure_output,
    query_from_state,
)
from agents.finance_agent.financial_query_agent.planner.models import (
    FinancialQueryPlan,
)
from agents.finance_agent.financial_query_agent.planner.prompts import (
    FINANCIAL_QUERY_PLANNER_PROMPT,
)
from agents.finance_agent.financial_query_agent.predefined.whitelist import (
    template_catalog_text,
)
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


def _normalize_plan(plan: FinancialQueryPlan, *, fallback: FinancialQueryPlan) -> FinancialQueryPlan:
    if plan.route not in {"predefined", "text_to_sql"}:
        return fallback
    if not plan.reason:
        plan.reason = fallback.reason
    return plan


async def _plan_with_llm(
    question: str,
    config: RunnableConfig = None,
) -> FinancialQueryPlan:
    fallback = _fallback_plan(question)
    if _looks_like_complex_query(question):
        return fallback

    try:
        llm = get_router_llm()
        plan = cast(
            FinancialQueryPlan,
            await llm.with_structured_output(
                FinancialQueryPlan,
                method="json_mode",
            ).ainvoke(
                [
                    ("system", FINANCIAL_QUERY_PLANNER_PROMPT),
                    (
                        "human",
                        f"用户问题：{question}\n\n白名单模板能力概览：\n{template_catalog_text()}",
                    ),
                ],
                config=config,
            ),
        )
    except Exception:
        logger.exception("financial_query_planner llm planning failed")
        return fallback

    return _normalize_plan(plan, fallback=fallback)


async def financial_query_planner(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """根据用户问题路由到 predefined 或 text_to_sql，不负责字段抽取。"""
    try:
        question = str(state.get("financial_query_text") or query_from_state(state)).strip()
    except ValueError:
        question = ""
    if not question:
        logger.error("financial_query_planner missing question")
        return database_failure_output(state, step="financial_query_planner_error")

    plan = await _plan_with_llm(question, config)
    logger.info(
        "financial_query_planner route={} reason={}",
        plan.route,
        plan.reason,
    )
    return {
        "financial_query_text": question,
        "financial_query_plan_route": plan.route,
        "financial_query_plan_reason": plan.reason,
        "steps": ["financial_query_planner"],
    }


__all__ = ["financial_query_planner"]
