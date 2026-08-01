"""Root Orchestrator 可调用的专业 Agent 注册表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agents.orchestrator.adapters import (
    agent_result_from_task_result,
    evidence_from_citation,
)
from agents.orchestrator.contracts import AgentResult, EvidencePolicy, TaskSpec
from agents.orchestrator.progress import AgentProgressJournal
from agents.runtime_context import AgentRuntimeContext


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Root Orchestrator 可调用处理器的稳定描述。"""

    agent_id: str
    description: str
    capabilities: tuple[str, ...]
    memory_keys: tuple[str, ...] = ()
    semantic_memory_types: tuple[str, ...] = ()
    kind: str = "agent"
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        agent_id="general_agent",
        description="处理无需外部事实依据的普通对话",
        capabilities=(),
        memory_keys=(
            "response_language",
            "response_detail_level",
            "preferred_output_format",
        ),
        semantic_memory_types=("episodic",),
    ),
    AgentSpec(
        agent_id="finance_agent",
        description="处理金融知识、财务数据库和 RAG 分析",
        capabilities=("faq", "pdf", "financial_query"),
        memory_keys=(
            "response_language",
            "response_detail_level",
            "preferred_output_format",
            "default_currency",
            "default_market",
            "default_compare_period",
            "citation_preference",
        ),
        semantic_memory_types=("episodic",),
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=1,
            require_provenance=True,
        ),
    ),
    AgentSpec(
        agent_id="market_acquisition_workflow",
        description="按明确的数据类型从受治理市场数据源采集结构化数据",
        capabilities=(
            "iwencai.market.query",
            "iwencai.industry.query",
            "iwencai.index.query",
            "iwencai.fund.screen",
        ),
        memory_keys=("default_currency", "default_market"),
        kind="workflow",
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=1,
            require_provenance=True,
            require_structured_data=True,
        ),
    ),
    AgentSpec(
        agent_id="research_retrieval_workflow",
        description="按明确的资料类型检索公告、研报和机构评级",
        capabilities=(
            "iwencai.announcement.search",
            "iwencai.report.search",
            "iwencai.rating.query",
        ),
        memory_keys=("response_language", "citation_preference"),
        kind="workflow",
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=1,
            require_provenance=True,
        ),
    ),
    AgentSpec(
        agent_id="stock_screening_agent",
        description="理解复杂自然语言选股条件并调用受治理选股工具",
        capabilities=("iwencai.screen",),
        memory_keys=(
            "response_language",
            "preferred_output_format",
            "default_currency",
            "default_market",
        ),
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=1,
            require_provenance=True,
            require_structured_data=True,
        ),
    ),
    AgentSpec(
        agent_id="market.compute",
        description="对上游 CandidateSet 执行确定性过滤、排序和截取",
        capabilities=("market.compute",),
        memory_keys=(
            "default_currency",
            "default_market",
            "default_compare_period",
        ),
        kind="deterministic",
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=1,
            require_provenance=True,
            require_structured_data=True,
        ),
    ),
    AgentSpec(
        agent_id="research_workflow",
        description="围绕一个问题完成多源采集、Deep Agent 分析和质量收敛",
        capabilities=("deep.research",),
        memory_keys=(
            "response_language",
            "response_detail_level",
            "preferred_output_format",
            "citation_preference",
        ),
        semantic_memory_types=("episodic",),
        kind="workflow",
        evidence_policy=EvidencePolicy(
            required=True,
            min_count=2,
            require_provenance=True,
        ),
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


def _dependency_payload(results: list[AgentResult]) -> list[AgentResult]:
    """复制完整上游结果，避免下游修改编排器持有的对象。"""
    return [item.model_copy(deep=True) for item in results]


async def invoke_agent(
    task: TaskSpec,
    *,
    dependency_results: list[AgentResult],
    memory_context: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    progress: AgentProgressJournal | None = None,
) -> AgentResult:
    """以统一契约调用现有专业 Agent。"""
    get_agent_spec(task.agent_id)
    query = task.objective
    invocation_state = {
        "messages": [HumanMessage(content=query)],
        "dependency_results": _dependency_payload(dependency_results),
        "task_input": dict(task.input_data),
        "task_identity": {
            "task_id": task.task_id,
            "logical_task_id": task.logical_task_id or task.task_id,
            "attempt_id": task.attempt_id,
            "attempt_number": task.attempt_number,
            "idempotency_key": task.idempotency_key,
        },
        # 调用边界只接收当前 task 的投影，禁止传入完整任务映射。
        "memory_context": deepcopy(memory_context or {}),
    }
    planning_scope = task.input_data.get("domain_planning_scope")
    if isinstance(planning_scope, dict):
        invocation_state["domain_planning_scope"] = dict(planning_scope)

    if task.agent_id == "market_acquisition_workflow":
        from agents.market_acquisition_workflow import (
            run_market_acquisition_workflow,
        )

        result = await run_market_acquisition_workflow(
            invocation_state,
            query=query,
            config=config,
            runtime=runtime,
        )
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "research_retrieval_workflow":
        from agents.research_retrieval_workflow import (
            run_research_retrieval_workflow,
        )

        result = await run_research_retrieval_workflow(
            invocation_state,
            query=query,
            config=config,
            runtime=runtime,
        )
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "stock_screening_agent":
        from agents.stock_screening_agent import run_stock_screening_agent

        result = await run_stock_screening_agent(
            invocation_state,
            query=query,
            config=config,
            runtime=runtime,
        )
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "market.compute":
        from agents.market_compute.executor import run_market_compute

        result = await run_market_compute(invocation_state, query=query)
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "research_workflow":
        from agents.research_workflow import run_research_workflow

        result = await run_research_workflow(
            invocation_state,
            query=query,
            config=config,
            runtime=runtime,
        )
        return result.model_copy(update={"task_id": task.task_id})

    if task.agent_id == "finance_agent":
        from agents.finance_agent import finance_agent

        if progress is None:
            output = await finance_agent.ainvoke(invocation_state, config=config)
        else:
            output: dict[str, Any] = {}
            async for snapshot in finance_agent.astream(
                invocation_state,
                config=config,
                stream_mode="values",
            ):
                if not isinstance(snapshot, dict):
                    continue
                output = snapshot
                progress.observe_finance_state(snapshot)
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
            invocation_state,
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


__all__ = [
    "AGENT_SPECS",
    "AgentSpec",
    "get_agent_spec",
    "invoke_agent",
    "list_agent_specs",
]
