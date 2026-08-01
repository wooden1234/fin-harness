"""General Agent 节点：闲聊 / 回溯 / 兜底，可调用受控工具（天气等）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.general_agent.prompts import GENERAL_BUSY_ANSWER, GENERAL_SYSTEM_PROMPT
from agents.general_agent.weather_direct import (
    format_weather_result,
    parse_weather_request,
)
from agents.llm import get_faq_llm
from agents.states import FinAgentState
from agents.runtime_context import AgentRuntimeContext
from agents.tool_runtime import run_with_tools
from tools.weather import fetch_weather

# General Agent 可绑定的工具（按 tool_id）
GENERAL_TOOL_IDS: tuple[str, ...] = ("weather.get",)


async def general_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
    query = next(
        (
            str(message.content or "")
            for message in reversed(list(state.get("messages") or []))
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    weather_request = parse_weather_request(query)
    if weather_request is not None:
        result = await fetch_weather(
            weather_request.city,
            days=weather_request.days,
        )
        return {"messages": [AIMessage(content=format_weather_result(result))]}
    return await run_with_tools(
        state,
        llm=get_faq_llm(),
        system_prompt=GENERAL_SYSTEM_PROMPT,
        tool_ids=GENERAL_TOOL_IDS,
        config=config,
        runtime=runtime,
        busy_answer=GENERAL_BUSY_ANSWER,
        agent_name="general_agent",
    )
