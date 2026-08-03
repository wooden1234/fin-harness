"""Root Graph 中的 Main DeepAgent 节点。"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from agents.context import conversation_messages
from agents.main_deep_agent.contracts import MainAgentResponse
from agents.main_deep_agent.assembly import run_main_deep_agent
from agents.main_deep_agent.middleware.compliance import (
    is_investment_action_sensitive,
)
from agents.orchestrator.analyzer import latest_query
from agents.orchestrator.contracts import AgentResult
from agents.orchestrator.state import OrchestratorState
from agents.runtime_context import AgentRuntimeContext
from app.core.config import settings


async def main_deep_agent_node(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """直接处理当前问题，不执行 Analyzer 或领域 Planner。"""
    context = runtime.context if runtime is not None else AgentRuntimeContext()
    query = latest_query(state)
    sensitive = is_investment_action_sensitive(query)
    response, journal, budget, status, detail = await run_main_deep_agent(
        messages=conversation_messages(state),
        context=context,
        config=config,
        investment_action_sensitive=sensitive,
    )
    if response is None and not journal.entries and detail and status == "completed":
        response = MainAgentResponse(mode="direct", direct_answer=detail)
    if response is None and status == "completed":
        status = "structured_output_failed"

    if budget.budget_tier == "deep_research":
        soft_seconds = float(settings.MAIN_AGENT_RESEARCH_SOFT_DEADLINE_SEC)
        hard_seconds = float(settings.MAIN_AGENT_RESEARCH_HARD_DEADLINE_SEC)
    elif budget.budget_tier == "standard":
        soft_seconds = float(settings.MAIN_AGENT_STANDARD_SOFT_DEADLINE_SEC)
        hard_seconds = float(settings.MAIN_AGENT_STANDARD_HARD_DEADLINE_SEC)
    else:
        soft_seconds = float(settings.MAIN_AGENT_DIRECT_HARD_DEADLINE_SEC)
        hard_seconds = soft_seconds
    context.configure_budget(
        budget_tier=budget.budget_tier,
        soft_seconds=soft_seconds,
        hard_seconds=hard_seconds,
        unit_timeouts={
            "deterministic": 3.0,
            "agent": 12.0,
            "workflow": 30.0,
            "tool_skill": 15.0,
        },
    )

    evidence = list(journal.evidence.values())
    result = AgentResult(
        task_id="main",
        logical_task_id="main",
        agent_id="main_deep_agent",
        status="completed" if status == "completed" else "partial" if evidence else "failed",
        answer="",
        evidence=evidence,
        gaps=[] if status == "completed" else [status],
        error_code="" if status == "completed" else detail or status,
        metadata={
            "budget_tier": budget.budget_tier,
            "tool_calls": budget.tool_calls,
            "source_families": sorted(budget.independent_source_families),
        },
    )
    journal_snapshot = journal.snapshot()
    journal_snapshot["model_rounds"] = budget.model_rounds
    journal_snapshot["agent_status"] = status
    journal_snapshot["soft_deadline_reached"] = any(
        entry.error == "soft_deadline_no_new_tools" for entry in journal.entries
    )
    journal_snapshot["finalization_reason"] = budget.finalization_reason
    journal_snapshot["finalization_started_at"] = budget.finalization_started_at
    return {
        "main_agent_response": response,
        "main_agent_journal": journal_snapshot,
        "investment_action_sensitive": sensitive,
        "agent_results": Overwrite([result]),
        "evidence": Overwrite(evidence),
        "citations": Overwrite([]),
        "execution_status": status,
        "route": "main",
        "steps": ["main_deep_agent:completed"],
    }


__all__ = ["main_deep_agent_node"]
