"""Deep Agent 驱动的 A 股选股 Agent。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.orchestrator.adapters import citation_from_evidence
from agents.orchestrator.contracts import AgentResult
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState
from agents.stock_screening_agent.deep_runtime import run_stock_screening_deep_agent
from agents.stock_screening_agent.spec import STOCK_SCREENING_SPEC


def _latest_user_query(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


async def run_stock_screening_agent(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> AgentResult:
    return await run_stock_screening_deep_agent(
        state,
        query=query,
        config=config,
        runtime=runtime,
    )


async def stock_screening_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    query = str(state.get("rewritten_query") or "").strip() or _latest_user_query(state)
    result = await run_stock_screening_agent(
        state,
        query=query,
        config=config,
        runtime=runtime,
    )
    return {
        "messages": [AIMessage(content=result.answer)],
        "summary": result.answer,
        # 保留完整 AgentResult，供编排器和后续节点读取结构化数据、状态与缺口。
        "agent_results": [result],
        "citations": [citation_from_evidence(item) for item in result.evidence],
    }


__all__ = [
    "STOCK_SCREENING_SPEC",
    "run_stock_screening_agent",
    "stock_screening_agent",
]
