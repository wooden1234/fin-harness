"""predefined 白名单槽位抽取节点。"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.states import FinAgentState
from agents.finance_agent.financial_query_agent.common import query_from_state
from agents.finance_agent.financial_query_agent.predefined.intent import (
    FinancialQueryIntent,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.models import (
    PredefinedSlotExtraction,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.normalizer import (
    build_predefined_query_intent,
)
from agents.finance_agent.financial_query_agent.predefined.extraction.prompts import (
    PREDEFINED_SLOT_EXTRACTION_PROMPT,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")


async def _extract_slots_with_llm(
    question: str,
    config: RunnableConfig = None,
) -> PredefinedSlotExtraction:
    llm = get_router_llm()
    return cast(
        PredefinedSlotExtraction,
        await llm.with_structured_output(
            PredefinedSlotExtraction,
            method="json_mode",
        ).ainvoke(
            [
                ("system", PREDEFINED_SLOT_EXTRACTION_PROMPT),
                ("human", f"用户问题：{question}"),
            ],
            config=config,
        ),
    )


def _fallback_slots(question: str) -> PredefinedSlotExtraction:
    return PredefinedSlotExtraction(
        companies=[question],
        years=[],
        metrics=[],
        operation="lookup",
    )


async def extract_predefined_slots(
    question: str,
    config: RunnableConfig = None,
) -> FinancialQueryIntent:
    """从用户问题抽取白名单执行所需槽位。"""
    try:
        slots = await _extract_slots_with_llm(question, config)
    except Exception:
        logger.exception("predefined slot extraction failed")
        slots = _fallback_slots(question)

    intent = build_predefined_query_intent(slots)
    logger.info(
        "predefined slots company={} year={} metric={} operation={}",
        intent.company,
        intent.year,
        intent.metric,
        intent.operation,
    )
    if intent.ambiguity:
        logger.info("predefined ambiguity={}", intent.ambiguity)
    return intent


async def resolve_predefined_query_context(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> tuple[str, FinancialQueryIntent, dict]:
    """确保 predefined 路径拥有 query 与白名单意图。"""
    query = str(state.get("financial_query_text") or query_from_state(state)).strip()
    intent = state.get("financial_query_intent")
    if isinstance(intent, FinancialQueryIntent):
        return query, intent, {}

    intent = await extract_predefined_slots(query, config)
    return query, intent, {
        "financial_query_text": query,
        "financial_query_intent": intent,
    }


__all__ = [
    "extract_predefined_slots",
    "resolve_predefined_query_context",
]
