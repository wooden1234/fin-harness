"""Root Orchestrator 可调用的专业 Agent 注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.orchestrator.adapters import (
    agent_result_from_task_result,
    evidence_from_citation,
)
from agents.orchestrator.contracts import AgentResult, TaskSpec
from agents.runtime_context import AgentRuntimeContext


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """专业 Agent 的稳定描述。"""

    agent_id: str
    description: str
    capabilities: tuple[str, ...]


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        agent_id="general_agent",
        description="处理无需外部事实依据的普通对话",
        capabilities=(),
    ),
    AgentSpec(
        agent_id="finance_agent",
        description="处理金融知识、财务、文档和公开信息研究",
        capabilities=("faq", "pdf", "financial_query", "web_search"),
    ),
    AgentSpec(
        agent_id="stock_screening_agent",
        description="处理自然语言 A 股选股",
        capabilities=("iwencai.screen",),
    ),
)

_SPEC_BY_ID = {item.agent_id: item for item in AGENT_SPECS}


def get_agent_spec(agent_id: str) -> AgentSpec:
    """获取已登记 Agent，未知名称立即失败。"""
    try:
        return _SPEC_BY_ID[agent_id]
    except KeyError as exc:
        raise ValueError(f"agent_not_registered:{agent_id}") from exc


def list_agent_specs() -> list[AgentSpec]:
    return list(AGENT_SPECS)


async def invoke_agent(
    task: TaskSpec,
    *,
    dependency_results: list[AgentResult],
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> AgentResult:
    """以统一契约调用现有专业 Agent。"""
    get_agent_spec(task.agent_id)
    query = task.objective
    if dependency_results:
        dependency_text = "\n\n".join(
            f"上游任务 {item.task_id} 结果：{item.answer}"
            for item in dependency_results
        )
        query = f"{query}\n\n{dependency_text}"

    if task.agent_id == "stock_screening_agent":
        from agents.stock_screening_agent import run_stock_screening_agent

        result = await run_stock_screening_agent(
            {"messages": [HumanMessage(content=query)]},
            query=query,
            config=config,
            runtime=runtime,
        )
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "finance_agent":
        from agents.finance_agent import finance_agent

        output = await finance_agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        unified_results = list(output.get("agent_results") or [])
        if unified_results:
            converted = [
                item.model_copy(update={"agent_id": "finance_agent"})
                if isinstance(item, AgentResult)
                else AgentResult.model_validate(item)
                for item in unified_results
            ]
            return _merge_output_evidence(
                task.task_id,
                converted,
                list(output.get("citations") or []),
            )

        results = list(output.get("task_results") or [])
        if results:
            converted = [
                agent_result_from_task_result(item, agent_id="finance_agent")
                for item in results
            ]
            return _merge_output_evidence(
                task.task_id,
                converted,
                list(output.get("citations") or []),
            )
        return AgentResult(
            task_id=task.task_id,
            agent_id="finance_agent",
            status="completed" if output.get("summary") else "uncovered",
            answer=str(output.get("summary") or ""),
            evidence=[
                evidence_from_citation(item, task_id=task.task_id)
                for item in list(output.get("citations") or [])
                if isinstance(item, dict)
            ],
        )

    if task.agent_id == "general_agent":
        from agents.general_agent import general_agent

        output = await general_agent(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            runtime=runtime,
        )
        return AgentResult(
            task_id=task.task_id,
            agent_id="general_agent",
            status="completed",
            answer=_last_message_text(output),
        )

    raise ValueError(f"agent_not_implemented:{task.agent_id}")


def _last_message_text(output: dict[str, Any]) -> str:
    messages = list(output.get("messages") or [])
    return str(messages[-1].content or "") if messages else str(output.get("summary") or "")


def _merge_results(task_id: str, results: list[AgentResult]) -> AgentResult:
    if not results:
        return AgentResult(task_id=task_id, agent_id="finance_agent", status="uncovered")
    rank = {"completed": 4, "partial": 3, "clarify": 2, "uncovered": 1, "failed": 0}
    best = max(results, key=lambda item: rank.get(item.status, 0))
    return best.model_copy(update={"task_id": task_id})


def _merge_output_evidence(
    task_id: str,
    results: list[AgentResult],
    citations: list[Any],
) -> AgentResult:
    """合并 Worker 与 Finance 顶层引用，避免适配边界丢失证据。"""
    best = _merge_results(task_id, results)
    existing_ids = {item.evidence_id for item in best.evidence}
    extra = [
        evidence_from_citation(item, task_id=task_id)
        for item in citations
        if isinstance(item, dict)
    ]
    merged = list(best.evidence)
    merged.extend(item for item in extra if item.evidence_id not in existing_ids)
    return best.model_copy(update={"task_id": task_id, "evidence": merged})


__all__ = ["AGENT_SPECS", "AgentSpec", "get_agent_spec", "invoke_agent", "list_agent_specs"]
