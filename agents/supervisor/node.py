"""Supervisor：选择普通对话、金融任务、改写或追问动作。"""

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

RouteTarget = Literal[
    "general_agent",
    "plan_agent",
    "query_rewrite",
    "final_answer",
    "error_handler",
]

_REWRITE_ATTEMPTED_STATUSES = frozenset(
    {"success", "passthrough", "uncertain", "fallback"}
)


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
    rewritten_query = str(state.get("rewritten_query") or "").strip()
    rewrite_status = str(state.get("rewrite_status") or "").strip()
    if rewritten_query and rewrite_status in _REWRITE_ATTEMPTED_STATUSES:
        messages.append(
            SystemMessage(content=f"[本轮改写后的问题]\n{rewritten_query}")
        )

    logger.info("----- Supervisor: analyze_and_route_query -----")
    logger.info("history_messages={}", len(messages) - 1)

    router = cast(
       Router,
       await model.with_structured_output(
            Router, method="json_mode"
        ).ainvoke(messages, config=config),
    )
    logger.info("action={} logic={}", router.action, router.logic)

    route = router.action if router.action in {"general", "plan"} else ""
    return {
        "route": route,
        "supervisor_action": router.action,
        "logic": router.logic,
    }


def route_query(state: FinAgentState) -> RouteTarget:
    """条件边：根据 Supervisor 的单一动作选择下一节点。"""
    action = str(state.get("supervisor_action") or state.get("route") or "")
    rewrite_status = str(state.get("rewrite_status") or "")
    rewrite_attempted = rewrite_status in _REWRITE_ATTEMPTED_STATUSES
    logger.info(
        "route_query: action={} rewrite_attempted={}",
        action,
        rewrite_attempted,
    )

    if action in {"rewrite", "clarify"}:
        return "final_answer" if rewrite_attempted else "query_rewrite"
    if action == "general":
        return "general_agent"
    if action == "plan":
        return "plan_agent"

    logger.warning("未知 supervisor_action={}，走错误兜底", action)
    return "error_handler"
