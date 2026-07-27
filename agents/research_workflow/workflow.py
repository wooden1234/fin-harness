"""独立金融研究工作流：采集证据、DeepAgent 研究、质量收敛。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.orchestrator.contracts import AgentResult, Evidence
from agents.research_workflow.contracts import ResearchPlan
from agents.research_workflow.planner import plan_research_adaptive
from agents.research_workflow.spec import RESEARCH_WORKFLOW_SPEC
from agents.research_workflow.state import ResearchWorkflowState
from agents.runtime_context import AgentRuntimeContext


async def plan_research(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """生成只属于 Research Workflow 的研究级计划。"""
    task_input = state.get("task_input")
    plan = await plan_research_adaptive(
        str(state.get("query") or "").strip(),
        task_input if isinstance(task_input, Mapping) else {},
        config=config,
    )
    return {
        "research_plan": plan,
        "source_tasks": list(plan.source_tasks),
    }


async def collect_research_sources(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """在研究工作流内部并行采集初始证据。"""
    plan = state.get("research_plan")
    tasks = list(plan.source_tasks) if plan is not None else []
    if not tasks:
        return {"source_tasks": [], "source_results": []}

    from agents.orchestrator.agent_registry import invoke_agent

    calls = [
        invoke_agent(
            task,
            dependency_results=[],
            config=config,
            runtime=runtime,
        )
        for task in tasks
    ]
    outputs = await asyncio.gather(*calls, return_exceptions=True)
    results: list[AgentResult] = []
    for task, output in zip(tasks, outputs, strict=True):
        if isinstance(output, BaseException):
            results.append(
                AgentResult(
                    task_id=task.task_id,
                    agent_id=task.agent_id,
                    status="failed",
                    error_code=type(output).__name__,
                    gaps=[str(output)],
                )
            )
        else:
            results.append(output)
    return {"source_tasks": tasks, "source_results": results}


async def run_deep_research(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """把来源任务和外部依赖统一交给受限 DeepAgent 分析。"""
    from agents.deep_research_agent import run_deep_research_agent

    plan = state.get("research_plan")
    dependencies = [
        *list(state.get("dependency_results") or []),
        *([_research_plan_result(plan)] if plan is not None else []),
        *list(state.get("source_results") or []),
    ]
    result = await run_deep_research_agent(
        {
            "messages": list(state.get("messages") or []),
            "dependency_results": dependencies,
        },
        query=str(state.get("query") or ""),
        config=config,
        runtime=runtime,
    )
    return {"deep_result": result}


def _research_plan_result(plan: ResearchPlan) -> AgentResult:
    """通过只读依赖通道把可信研究计划交给 DeepAgent。"""
    return AgentResult(
        task_id="research:plan",
        agent_id="research_workflow.planner",
        status="completed",
        answer="按 Research Workflow 生成的结构化研究计划执行。",
        structured_data=plan.model_dump(mode="json"),
        metadata={"result_type": "research_plan"},
    )


def _merge_evidence(results: Sequence[AgentResult]) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for result in results:
        for item in result.evidence:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                evidence.append(item)
    return evidence


async def finalize_research(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """统一收敛内部执行结果，对 Root 只暴露一个研究结果。"""
    del config
    deep_result = state.get("deep_result")
    if deep_result is None:
        result = AgentResult(
            task_id=RESEARCH_WORKFLOW_SPEC.default_task_id,
            agent_id=RESEARCH_WORKFLOW_SPEC.workflow_id,
            status="failed",
            answer="研究工作流未生成研究结论。",
            error_code="research_deep_result_missing",
            gaps=["Deep Research Agent 未返回结果"],
        )
        return {"result": result}

    source_results = list(state.get("source_results") or [])
    research_plan = state.get("research_plan")
    evidence = _merge_evidence([*source_results, deep_result])
    source_issues = [
        item
        for item in source_results
        if item.status in {"failed", "uncovered", "clarify"}
    ]
    gaps = list(deep_result.gaps)
    for item in source_issues:
        details = item.gaps or [item.error_code or item.status]
        gaps.extend(f"{item.task_id}: {detail}" for detail in details)
    gaps = list(dict.fromkeys(gaps))

    status = deep_result.status
    error_code = deep_result.error_code
    if status == "completed" and source_issues:
        status = "partial"
        error_code = "research_sources_partial"

    result = AgentResult(
        task_id=RESEARCH_WORKFLOW_SPEC.default_task_id,
        agent_id=RESEARCH_WORKFLOW_SPEC.workflow_id,
        status=status,
        answer=deep_result.answer,
        structured_data=dict(deep_result.structured_data),
        evidence=evidence,
        gaps=gaps,
        suggested_tasks=list(deep_result.suggested_tasks),
        error_code=error_code,
        metadata={
            "runtime": "research_workflow",
            "deep_agent_id": deep_result.agent_id,
            "research_plan": (
                research_plan.model_dump(mode="json")
                if research_plan is not None
                else {}
            ),
            "source_task_ids": [item.task_id for item in source_results],
            "source_results": [
                item.model_dump(mode="json") for item in source_results
            ],
            "deep_agent_metadata": dict(deep_result.metadata),
        },
    )
    return {"result": result}


def build_research_workflow() -> StateGraph:
    """构建独立研究子图，DeepAgent 只是其中的研究节点。"""
    builder = StateGraph(
        ResearchWorkflowState,
        context_schema=AgentRuntimeContext,
    )
    builder.add_node("plan_research", plan_research)
    builder.add_node("collect_sources", collect_research_sources)
    builder.add_node("deep_research", run_deep_research)
    builder.add_node("finalize_research", finalize_research)
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "collect_sources")
    builder.add_edge("collect_sources", "deep_research")
    builder.add_edge("deep_research", "finalize_research")
    builder.add_edge("finalize_research", END)
    return builder


@lru_cache(maxsize=1)
def get_research_workflow():
    return build_research_workflow().compile(name=RESEARCH_WORKFLOW_SPEC.workflow_id)


async def run_research_workflow(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> AgentResult:
    """运行完整研究子图并返回单一标准结果。"""
    output = await get_research_workflow().ainvoke(
        {
            "messages": list(state.get("messages") or []),
            "query": query,
            "task_input": dict(state.get("task_input") or {}),
            "dependency_results": list(state.get("dependency_results") or []),
        },
        config=config,
        context=runtime.context if runtime is not None else None,
    )
    result = output.get("result")
    if isinstance(result, AgentResult):
        return result
    return AgentResult.model_validate(result)


__all__ = [
    "build_research_workflow",
    "get_research_workflow",
    "run_research_workflow",
]
