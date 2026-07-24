"""finance_agent planner 共享工具。"""

from __future__ import annotations

import uuid
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.states import PlannerOutput, SubTask
from agents.finance_agent.planner.prompts import (
    PLANNER_REPAIR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from agents.turn_workspace import reset_worker_workspace
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
    """兼容旧名：plan 路径只重置 worker 输出，不 Overwrite steps。

    完整临时工作区重置见 ``agents.turn_workspace.begin_turn_workspace``（init_turn 入口）。
    """
    return reset_worker_workspace()


def assign_task_ids(tasks: list[SubTask]) -> list[SubTask]:
    for task in tasks:
        task.id = uuid.uuid4().hex[:8]
    return tasks


def empty_plan(*, step: str, reason: str) -> dict:
    logger.warning("planner empty_plan step={} reason={}", step, reason)
    return {
        **reset_worker_workspace(),
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
    human_prompt: str,
    config: RunnableConfig | None,
) -> PlannerOutput:
    """API/超时类错误重试一次；仍失败则抛出。"""
    try:
        return await ainvoke_planner(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            human_prompt=human_prompt,
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
            human_prompt=human_prompt,
            config=config,
        )


async def repair_plan(
    query: str,
    raw_tasks: list[SubTask],
    issues: list[str],
    config: RunnableConfig | None,
    *,
    conversation_summary: str = "",
    rewritten_query: str = "",
) -> PlannerOutput:
    payload = PlannerOutput(tasks=raw_tasks).model_dump_json()
    parts = [
        f"此前对话摘要：\n{conversation_summary.strip() or '无'}",
        f"当前用户问题（原文）：\n{query}",
    ]
    rewritten = rewritten_query.strip()
    if rewritten and rewritten != query.strip():
        parts.append(f"改写后的完整问题：\n{rewritten}")
    parts.append(f"校验问题：{', '.join(issues)}")
    parts.append(f"待修正输出：{payload}")
    return await ainvoke_planner(
        system_prompt=PLANNER_REPAIR_SYSTEM_PROMPT,
        human_prompt="\n\n".join(parts),
        config=config,
    )


__all__ = [
    "CLARIFICATION_ANSWER",
    "assign_task_ids",
    "begin_turn_workspace",
    "empty_plan",
    "is_schema_error",
    "is_transient_api_error",
    "latest_user_query",
    "logger",
    "plan_with_retry",
    "repair_plan",
]
