"""Deep Research Agent 的 v2 调用入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.deep_research_agent.deep_runtime import run_deep_research_agent
from agents.deep_research_agent.spec import DEEP_RESEARCH_SPEC
from agents.orchestrator.adapters import citation_from_evidence
from agents.orchestrator.contracts import AgentResult
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState


def _latest_user_query(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


async def deep_research_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    query = str(state.get("rewritten_query") or "").strip() or _latest_user_query(state)
    result = await run_deep_research_agent(
        state,
        query=query,
        config=config,
        runtime=runtime,
    )
    return {
        "messages": [AIMessage(content=result.answer)],
        "summary": result.answer,
        "agent_results": [result],
        "citations": [citation_from_evidence(item) for item in result.evidence],
    }


__all__ = [
    "DEEP_RESEARCH_SPEC",
    "deep_research_agent",
    "run_deep_research_agent",
]
