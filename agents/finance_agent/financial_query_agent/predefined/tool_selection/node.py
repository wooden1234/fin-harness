"""predefined 工具选择节点，对应 assistgen tool_selection。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.finance_agent.financial_query_agent.predefined.extraction.models import (
    PredefinedSlotExtraction,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection.models import (
    is_valid_template_id,
    predefined_sql,
)
from agents.finance_agent.financial_query_agent.predefined.tool_selection.prompts import (
    PREDEFINED_TOOL_SELECTION_PROMPT,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")


@dataclass(frozen=True)
class PredefinedToolSelectionResult:
    """tool_selection 的显式结果，失败时由上层转入复杂查询。"""

    success: bool
    template_id: str
    slots: PredefinedSlotExtraction | None
    error: str = ""


def _operation_for_template(template_id: str) -> str:
    mapping = {
        "exact_metric_lookup": "lookup",
        "latest_metric_lookup": "latest",
        "compare_metric_lookup": "compare",
        "trend_metric_lookup": "trend",
    }
    return mapping.get(template_id, "lookup")


def _tool_call_to_slots(tool_call: predefined_sql) -> PredefinedSlotExtraction:
    return PredefinedSlotExtraction(
        companies=list(tool_call.companies),
        years=list(tool_call.years),
        metrics=list(tool_call.metrics),
        operation=_operation_for_template(tool_call.template_id),
        top_k=tool_call.top_k,
    )


async def select_predefined_tool(
    question: str,
    config: RunnableConfig = None,
) -> PredefinedToolSelectionResult:
    """LLM 选择白名单模板并提取参数，失败时返回可兜底的错误状态。"""
    llm = get_router_llm()
    chain = (
        llm.bind_tools(tools=[predefined_sql])
        | PydanticToolsParser(tools=[predefined_sql], first_tool_only=True)
    )
    try:
        tool_call = cast(
            predefined_sql,
            await chain.ainvoke(
                [
                    ("system", PREDEFINED_TOOL_SELECTION_PROMPT),
                    ("human", f"Question: {question}"),
                ],
                config=config,
            ),
        )
    except Exception:
        logger.exception("predefined tool_selection failed")
        return PredefinedToolSelectionResult(
            success=False,
            template_id="",
            slots=None,
            error="tool_selection_failed",
        )

    if not isinstance(tool_call, predefined_sql):
        logger.error("predefined tool_selection returned no predefined_sql call")
        return PredefinedToolSelectionResult(
            success=False,
            template_id="",
            slots=None,
            error="tool_selection_missing_tool_call",
        )

    template_id = tool_call.template_id
    if not is_valid_template_id(template_id):
        logger.error("predefined tool_selection invalid template_id={}", template_id)
        return PredefinedToolSelectionResult(
            success=False,
            template_id=template_id,
            slots=None,
            error="tool_selection_invalid_template",
        )

    slots = _tool_call_to_slots(tool_call)
    logger.info(
        "predefined tool_selection template_id={} raw_companies={} raw_years={} raw_metrics={}",
        template_id,
        slots.companies,
        slots.years,
        slots.metrics,
    )
    return PredefinedToolSelectionResult(
        success=True,
        template_id=template_id,
        slots=slots,
        error="",
    )


__all__ = ["PredefinedToolSelectionResult", "select_predefined_tool"]
