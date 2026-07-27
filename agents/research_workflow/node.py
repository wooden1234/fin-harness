"""独立金融研究工作流的 LangGraph 节点入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.orchestrator.adapters import citation_from_evidence
from agents.research_workflow.workflow import run_research_workflow
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState


def _latest_user_query(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


async def research_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    query = str(state.get("rewritten_query") or "").strip() or _latest_user_query(state)
    result = await run_research_workflow(
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


__all__ = ["research_workflow", "run_research_workflow"]
