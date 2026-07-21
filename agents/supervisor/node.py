"""Supervisor：意图分类（仅分类，风险已拆至 risk_triage）。"""

from __future__ import annotations

from typing import Literal, cast

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.context import conversation_messages
from agents.llm import get_router_llm
from agents.supervisor.prompts import SUPERVISOR_SYSTEM_PROMPT
from agents.states import FinAgentState, Router
from app.core.logger import get_logger

logger = get_logger(service="supervisor")

RouteTarget = Literal["general_agent", "risk_triage", "error_handler"]


async def analyze_and_route_query(
    state: FinAgentState,
    config: RunnableConfig,
) -> dict[str, str]:
    """分析用户问题，写入 route / logic。

    使用 ``with_structured_output(Router)`` 约束 LLM 输出。
    """
    model = get_router_llm()
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        *conversation_messages(state),
    ]

    logger.info("----- Supervisor: analyze_and_route_query -----")
    logger.info("history_messages={}", len(messages) - 1)

    router = cast(
       Router,
       await model.with_structured_output(
            Router, method="json_mode"
        ).ainvoke(messages, config=config),
    )
    logger.info("route={} logic={}", router.type, router.logic)

    return {"route": router.type, "logic": router.logic}


def route_query(state: FinAgentState) -> RouteTarget:
    """条件边：根据 Supervisor 的 route 选择下一节点。"""
    route = state.get("route", "general")
    logger.info("route_query: route={}", route)

    if route == "general":
        return "general_agent"
    if route == "plan":
        return "risk_triage"

    logger.warning("未知 route={}，走错误兜底", route)
    return "error_handler"
