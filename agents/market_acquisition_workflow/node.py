"""确定性市场数据采集工作流。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.iwencai_source_runtime import run_iwencai_source_tool
from agents.market_acquisition_workflow.spec import (
    MARKET_ACQUISITION_WORKFLOW_SPEC,
)
from agents.orchestrator.adapters import citation_from_evidence
from agents.orchestrator.contracts import AgentResult
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentState


def _latest_user_query(state: Mapping[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


async def run_market_acquisition_workflow(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> AgentResult:
    """按已确定的 Tool 执行采集，不进行工具选择或 Agent 委派。"""
    del config
    task_input = state.get("task_input")
    selected_tool = (
        str(task_input.get("market_tool_id") or "").strip()
        if isinstance(task_input, Mapping)
        else ""
    )
    if not selected_tool:
        return AgentResult(
            task_id=MARKET_ACQUISITION_WORKFLOW_SPEC.default_task_id,
            agent_id=MARKET_ACQUISITION_WORKFLOW_SPEC.workflow_id,
            status="clarify",
            answer="请明确需要查询行情、指数、行业还是基金数据。",
            error_code="market_source_missing",
            gaps=["task_input.market_tool_id 未提供"],
        )
    if selected_tool not in MARKET_ACQUISITION_WORKFLOW_SPEC.tool_ids:
        return AgentResult(
            task_id=MARKET_ACQUISITION_WORKFLOW_SPEC.default_task_id,
            agent_id=MARKET_ACQUISITION_WORKFLOW_SPEC.workflow_id,
            status="failed",
            error_code="market_acquisition_tool_not_allowed",
            gaps=[f"市场采集工作流不允许使用 Tool：{selected_tool}"],
        )

    skill_name = MARKET_ACQUISITION_WORKFLOW_SPEC.skill_for_tool(selected_tool)
    if skill_name is None:
        return AgentResult(
            task_id=MARKET_ACQUISITION_WORKFLOW_SPEC.default_task_id,
            agent_id=MARKET_ACQUISITION_WORKFLOW_SPEC.workflow_id,
            status="failed",
            error_code="market_acquisition_skill_missing",
            gaps=[f"市场数据 Tool 未绑定 Skill：{selected_tool}"],
        )
    return await run_iwencai_source_tool(
        tool_id=selected_tool,
        skill_name=skill_name,
        query=query,
        task_input=task_input if isinstance(task_input, Mapping) else {},
        runtime=runtime,
        agent_id=MARKET_ACQUISITION_WORKFLOW_SPEC.workflow_id,
        task_id=MARKET_ACQUISITION_WORKFLOW_SPEC.default_task_id,
    )


async def market_acquisition_workflow(
    state: FinAgentState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    query = str(state.get("rewritten_query") or "").strip() or _latest_user_query(state)
    result = await run_market_acquisition_workflow(
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
    "MARKET_ACQUISITION_WORKFLOW_SPEC",
    "market_acquisition_workflow",
    "run_market_acquisition_workflow",
]
