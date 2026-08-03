"""General Agent 节点：闲聊 / 回溯 / 天气等，统一走短 prompt + 受控工具。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.general_agent.prompts import GENERAL_BUSY_ANSWER, GENERAL_SYSTEM_PROMPT
from agents.llm import get_faq_llm
from agents.states import FinAgentState
from agents.runtime_context import AgentRuntimeContext
from agents.tool_runtime import run_with_tools

# General Agent 可绑定的工具（按 tool_id）
GENERAL_TOOL_IDS: tuple[str, ...] = ("weather.get",)


async def general_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict:
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
