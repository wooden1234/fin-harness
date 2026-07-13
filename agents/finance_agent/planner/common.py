"""finance_agent planner 共享工具。"""

from __future__ import annotations

import uuid
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from agents.llm import get_router_llm
from agents.states import PlannerOutput, SubTask
from agents.finance_agent.planner.prompts import (
    PLANNER_REPAIR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from app.core.logger import get_logger

logger = get_logger(service="finance_agent_supervisor")

CLARIFICATION_ANSWER = (
    "抱歉，我暂时无法判断这个问题需要查询哪类金融资料。"
    "请补充更明确的金融对象或查询目标，例如交易规则、产品费率、报告名称、公司、年份或财务指标。"
)


def latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def begin_turn_workspace() -> dict:
    """每轮用户提问开始时清空中间工作区，避免 checkpoint 跨轮污染。"""
    return {
        "task_results": Overwrite([]),
        "citations": Overwrite([]),
        "summary": "",
    }


def assign_task_ids(tasks: list[SubTask]) -> list[SubTask]:
    for task in tasks:
        task.id = uuid.uuid4().hex[:8]
    return tasks


def empty_plan(*, step: str, reason: str) -> dict:
    logger.warning("planner empty_plan step={} reason={}", step, reason)
    return {
        **begin_turn_workspace(),
        "sub_tasks": [],
        "steps": [f"{step}:{reason}"],
    }


def is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    name = type(exc).__name__.lower()
    markers = (
        "timeout",
        "connection",
        "ratelimit",
        "rate_limit",
        "apiconnection",
        "internalserver",
        "serviceunavailable",
        "429",
        "502",
        "503",
    )
    return any(marker in name for marker in markers)


def is_schema_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    markers = (
        "validation",
        "json",
        "parse",
        "outputparser",
        "structuredoutput",
        "pydantic",
        "badrequest",
    )
    return any(marker in name for marker in markers)


async def ainvoke_planner(
    *,
    system_prompt: str,
    human_prompt: str,
    config: RunnableConfig | None,
) -> PlannerOutput:
    llm = get_router_llm()
    return cast(
        PlannerOutput,
        await llm.with_structured_output(
            PlannerOutput, method="json_mode"
        ).ainvoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ],
            config=config,
        ),
    )


async def plan_with_retry(
    query: str,
    config: RunnableConfig | None,
) -> PlannerOutput:
    """API/超时类错误重试一次；仍失败则抛出。"""
    human = f"用户问题：{query}"
    try:
        return await ainvoke_planner(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            human_prompt=human,
            config=config,
        )
    except Exception as exc:
        if not is_transient_api_error(exc):
            raise
        logger.warning(
            "planner transient api error, retrying once: {}",
            type(exc).__name__,
        )
        return await ainvoke_planner(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            human_prompt=human,
            config=config,
        )


async def repair_plan(
    query: str,
    raw_tasks: list[SubTask],
    issues: list[str],
    config: RunnableConfig | None,
) -> PlannerOutput:
    payload = PlannerOutput(tasks=raw_tasks).model_dump_json()
    human = (
        f"用户问题：{query}\n"
        f"校验问题：{', '.join(issues)}\n"
        f"待修正输出：{payload}"
    )
    return await ainvoke_planner(
        system_prompt=PLANNER_REPAIR_SYSTEM_PROMPT,
        human_prompt=human,
        config=config,
    )


__all__ = [
    "CLARIFICATION_ANSWER",
    "ainvoke_planner",
    "assign_task_ids",
    "begin_turn_workspace",
    "empty_plan",
    "get_router_llm",
    "is_schema_error",
    "is_transient_api_error",
    "latest_user_query",
    "logger",
    "plan_with_retry",
    "repair_plan",
]
