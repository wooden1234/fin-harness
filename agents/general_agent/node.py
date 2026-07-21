"""General Agent 节点：纯 LLM 对话（闲聊 / 回溯 / 兜底）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.context import conversation_messages
from agents.llm import get_faq_llm
from agents.general_agent.prompts import GENERAL_BUSY_ANSWER, GENERAL_SYSTEM_PROMPT
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="general_agent")


async def general_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    llm_messages = [
        SystemMessage(content=GENERAL_SYSTEM_PROMPT),
        *conversation_messages(state),
    ]
    logger.info("general_agent history_messages={}", len(llm_messages) - 1)

    try:
        llm = get_faq_llm()
        parts: list[str] = []
        async for chunk in llm.astream(llm_messages, config=config):
            if chunk.content:
                parts.append(
                    chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                )
        answer = "".join(parts)
    except Exception:
        logger.exception("general_agent llm invoke failed")
        return {"messages": [AIMessage(content=GENERAL_BUSY_ANSWER)]}

    return {"messages": [AIMessage(content=answer)]}
