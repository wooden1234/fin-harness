"""text_to_sql 信息不足追问中间件。"""

from __future__ import annotations

import re
from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.predefined.semantic.registry_seed import (
    GLOBAL_ALIASES,
)
from agents.finance_agent.financial_query_agent.services.entity_resolver import (
    EntityResolver,
)
from agents.llm import get_router_llm
from agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.middleware.base import (
    MiddlewareResult,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlState,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

FINANCIAL_QUERY_TEXT_TO_SQL_CLARIFICATION_PROMPT = """你是复杂结构化查询补问助手。请根据用户问题和当前缺失字段，生成一句简洁中文追问。

要求：
1. 只追问当前仍不足以生成 SQL 的关键信息
2. 一句话即可，不解释系统实现
3. 优先点名需要补充的字段或口径
"""

FINANCIAL_QUERY_TEXT_TO_SQL_NEEDS_CLARIFICATION_ANSWER = (
    "当前问题中的查询条件还不够明确。请补充更具体的公司名称、财务指标、统计年份或计算口径。"
)

_VAGUE_QUESTION_MARKERS = ("怎么样", "如何", "什么情况", "表现如何", "好不好")
_LATEST_MARKERS = ("最新", "最近", "当前", "现在")
_RANGE_MARKERS = ("趋势", "历年", "近三年", "近五年", "近两年", "近几年", "变化")
_RANKING_MARKERS = ("排名", "最高", "最低", "前十", "前五", "top")
_METADATA_MARKERS = ("有哪些公司", "有哪些指标", "指标列表", "公司列表", "映射", "字典", "文档列表")
_FINANCIAL_MARKERS = ("财务", "业绩", "收入", "利润", "现金流", "研发", "营收", "年报", "同比", "对比", "趋势")
_YEAR_RE = re.compile(r"20\d{2}")


def _fallback_clarification(missing_fields: list[str]) -> str:
    if missing_fields:
        field_names = {
            "company": "公司名称",
            "metric": "财务指标",
            "year": "统计年份",
            "years": "统计年份",
            "period": "统计周期",
            "scope": "统计口径",
            "calculation": "计算方式",
        }
        labels = [field_names.get(field, field) for field in missing_fields]
        return f"请补充更明确的{'、'.join(labels)}，我再继续生成查询。"
    return FINANCIAL_QUERY_TEXT_TO_SQL_NEEDS_CLARIFICATION_ANSWER


def _is_vague_question(question: str) -> bool:
    normalized = question.strip()
    if len(normalized) < 4:
        return True
    return any(marker in normalized for marker in _VAGUE_QUESTION_MARKERS) and len(normalized) < 12


def _contains_any(question: str, markers: tuple[str, ...]) -> bool:
    return any(marker.lower() in question.lower() for marker in markers)


def _has_company(question: str) -> bool:
    normalized = question.lower()
    for aliases in EntityResolver.COMPANY_ALIASES.values():
        if any(alias.lower() in normalized for alias in aliases):
            return True
    return False


def _has_metric(question: str) -> bool:
    normalized = question.replace(" ", "").lower()
    for alias in GLOBAL_ALIASES:
        if alias.replace(" ", "").lower() in normalized:
            return True
    for aliases in EntityResolver.METRIC_ALIASES.values():
        if any(alias.replace(" ", "").lower() in normalized for alias in aliases):
            return True
    return False


def _is_metadata_question(question: str) -> bool:
    return _contains_any(question, _METADATA_MARKERS)


def _is_global_query(question: str) -> bool:
    return _contains_any(question, _RANKING_MARKERS)


def _infer_missing_fields(question: str) -> list[str]:
    normalized = question.strip()
    if not normalized or _is_metadata_question(normalized):
        return []

    has_financial_marker = _contains_any(normalized, _FINANCIAL_MARKERS)
    has_metric = _has_metric(normalized)
    has_company = _has_company(normalized)
    has_year = bool(_YEAR_RE.search(normalized))
    has_latest_or_range = _contains_any(normalized, _LATEST_MARKERS) or _contains_any(
        normalized,
        _RANGE_MARKERS,
    )

    missing: list[str] = []
    if has_financial_marker and not has_metric:
        missing.append("metric")
    if has_metric and not has_company and not _is_global_query(normalized):
        missing.append("company")
    if has_metric and not has_year and not has_latest_or_range:
        missing.append("year")
    if "利润" in normalized and not any(term in normalized for term in ("净利润", "归母", "营业利润")):
        missing.append("scope")
    return list(dict.fromkeys(missing))


async def _build_clarification_answer(
    *,
    question: str,
    missing_fields: list[str],
    route_reason: str,
    config: RunnableConfig | None = None,
) -> str:
    fallback = _fallback_clarification(missing_fields)
    try:
        llm = get_router_llm()
        result = cast(
            str,
            await llm.ainvoke(
                [
                    ("system", FINANCIAL_QUERY_TEXT_TO_SQL_CLARIFICATION_PROMPT),
                    (
                        "human",
                        f"用户问题：{question}\n缺失字段：{missing_fields}\n原因：{route_reason}",
                    ),
                ],
                config=config,
            ),
        )
        content = getattr(result, "content", result)
        return str(content).strip() or fallback
    except Exception:
        logger.exception("text_to_sql clarification middleware failed")
        return fallback


class ClarificationMiddleware:
    """模糊问题或生成阶段判定信息不足时，截断流程并返回追问。"""

    async def before_generate(
        self,
        state: TextToSqlState,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        del config
        question = state["question"].strip()
        missing_fields = _infer_missing_fields(question)
        if not _is_vague_question(question) and not missing_fields:
            return None
        if not missing_fields:
            missing_fields = ["company", "metric", "year"]
        return MiddlewareResult(
            halt=True,
            halt_reason="clarify",
            halt_answer=_fallback_clarification(missing_fields),
            state_updates={
                "missing_fields": missing_fields,
                "route_reason": "问题缺少生成安全 SQL 所需的关键信息。",
            },
        )

    async def _clarify_from_generation(
        self,
        state: TextToSqlState,
        generated: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        if generated.route != "clarify":
            return None
        answer = await _build_clarification_answer(
            question=state["question"],
            missing_fields=list(generated.missing_fields),
            route_reason=generated.reason,
            config=config,
        )
        return MiddlewareResult(
            halt=True,
            halt_reason="clarify",
            halt_answer=answer,
            state_updates={
                "missing_fields": list(generated.missing_fields),
                "route_reason": generated.reason,
            },
        )

    async def after_generate(
        self,
        state: TextToSqlState,
        generated: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        return await self._clarify_from_generation(state, generated, config)

    async def after_correct(
        self,
        state: TextToSqlState,
        corrected: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        return await self._clarify_from_generation(state, corrected, config)


__all__ = ["ClarificationMiddleware"]
