"""独立金融研究工作流：采集证据、内部 DeepAgent 研究、质量收敛。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.orchestrator.contracts import AgentResult, Evidence
from agents.research_workflow.contracts import (
    QuestionEvidenceAssessment,
    ResearchPlan,
    ResearchQuestion,
)
from agents.research_workflow.planner import (
    _validate_source_tasks,
    plan_research_adaptive,
)
from agents.research_workflow.spec import RESEARCH_WORKFLOW_SPEC
from agents.research_workflow.state import ResearchWorkflowState
from agents.runtime_context import (
    AgentRuntimeContext,
    RunHardDeadlineExceeded,
    RunSoftDeadlineExceeded,
)
from app.core.config import settings


async def plan_research(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """生成只属于 Research Workflow 的研究级计划。"""
    task_input = state.get("task_input")
    plan = await plan_research_adaptive(
        str(state.get("query") or "").strip(),
        task_input if isinstance(task_input, Mapping) else {},
        task_identity=(
            state.get("task_identity")
            if isinstance(state.get("task_identity"), Mapping)
            else {}
        ),
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
    tasks = _validate_source_tasks(
        plan.query,
        tasks,
        validation_scope=str(
            plan.metadata.get("task_validation_scope") or "research:direct"
        ),
    )

    from agents.orchestrator.agent_registry import get_agent_spec, invoke_agent

    semaphore = asyncio.Semaphore(
        max(1, int(settings.AGENT_V2_MAX_CONCURRENCY))
    )

    async def _invoke_source(task):
        context = runtime.context if runtime is not None else None
        agent_kind = get_agent_spec(task.agent_id).kind
        configured_timeout = {
            "deterministic": float(settings.AGENT_V2_DETERMINISTIC_TIMEOUT_SEC),
            "workflow": float(settings.AGENT_V2_WORKFLOW_TIMEOUT_SEC),
        }.get(
            agent_kind,
            float(settings.AGENT_V2_AGENT_TIMEOUT_SEC),
        )
        if context is not None:
            timeout_seconds, timeout_limit = context.execution_timeout_for(
                agent_kind,
                default_seconds=configured_timeout,
            )
        else:
            timeout_seconds, timeout_limit = configured_timeout, "unit"
        if timeout_seconds <= 0:
            if timeout_limit == "hard":
                raise RunHardDeadlineExceeded("run_hard_deadline_exceeded")
            if timeout_limit == "soft":
                raise RunSoftDeadlineExceeded("run_soft_deadline_exceeded")
            raise TimeoutError("task_timeout")
        try:
            async with asyncio.timeout(timeout_seconds):
                async with semaphore:
                    result = await invoke_agent(
                        task,
                        dependency_results=[],
                        config=config,
                        runtime=runtime,
                    )
                    return result.model_copy(
                        update={
                            "logical_task_id": (
                                task.logical_task_id or task.task_id
                            ),
                            "attempt_id": task.attempt_id,
                            "attempt_number": task.attempt_number,
                            "idempotency_key": task.idempotency_key,
                            # 来源 Workflow 可能保留自身默认 task_id；进入
                            # Research 边界后统一归属到已校验的 Source Task。
                            "evidence": [
                                item.model_copy(
                                    update={"task_id": task.task_id}
                                )
                                for item in result.evidence
                            ],
                        }
                    )
        except TimeoutError as exc:
            if timeout_limit == "hard":
                raise RunHardDeadlineExceeded(
                    "run_hard_deadline_exceeded"
                ) from exc
            if timeout_limit == "soft":
                raise RunSoftDeadlineExceeded(
                    "run_soft_deadline_exceeded"
                ) from exc
            raise

    calls = [_invoke_source(task) for task in tasks]
    outputs = await asyncio.gather(*calls, return_exceptions=True)
    results: list[AgentResult] = []
    for task, output in zip(tasks, outputs, strict=True):
        if isinstance(output, asyncio.CancelledError):
            raise output
        if isinstance(output, RunHardDeadlineExceeded):
            raise output
        if isinstance(output, RunSoftDeadlineExceeded):
            raise output
        if isinstance(output, BaseException):
            error_code = (
                "task_timeout"
                if isinstance(output, TimeoutError)
                else type(output).__name__
            )
            results.append(
                AgentResult(
                    task_id=task.task_id,
                    logical_task_id=task.logical_task_id,
                    attempt_id=task.attempt_id,
                    attempt_number=task.attempt_number,
                    idempotency_key=task.idempotency_key,
                    agent_id=task.agent_id,
                    status="failed",
                    error_code=error_code,
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
    from agents.research_workflow.deep_agent import run_deep_research_agent

    plan = state.get("research_plan")
    if plan is None or bool(plan.metadata.get("scope_blocked")):
        return {
            "deep_result": AgentResult(
                task_id="deep-research",
                agent_id="deep_research_agent",
                status="uncovered",
                answer="研究范围为空，未执行来源采集或深度研究。",
                error_code="research_scope_empty",
                gaps=["Root 明确限制了可用研究来源，无法形成证据链。"],
            )
        }
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


def _has_provenance(item: Evidence) -> bool:
    """使用与 Root 质量门一致的最低来源可追溯标准。"""
    return bool(
        item.source_type
        and (
            item.provider
            or item.title
            or item.url
            or item.metadata
        )
    )


def _question_evidence(
    question: ResearchQuestion,
    source_results: Sequence[AgentResult],
) -> tuple[list[Evidence], list[AgentResult]]:
    """只选择该问题授权来源任务产生的有效结果，禁止跨问题借用证据。"""
    source_task_ids = set(question.source_task_ids)
    relevant_results = [
        result
        for result in source_results
        if result.task_id in source_task_ids
        and result.status in {"completed", "partial"}
    ]
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for result in relevant_results:
        for item in result.evidence:
            if item.task_id and item.task_id not in source_task_ids:
                continue
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            evidence.append(item)
    return evidence, relevant_results


def _assess_question(
    question: ResearchQuestion,
    source_results: Sequence[AgentResult],
) -> QuestionEvidenceAssessment:
    """执行单个 ResearchQuestion 的 EvidencePolicy。"""
    policy = question.evidence_policy
    evidence, relevant_results = _question_evidence(
        question,
        source_results,
    )
    required_count = max(policy.min_count, 1) if policy.required else 0
    missing_provenance_count = (
        sum(1 for item in evidence if not _has_provenance(item))
        if policy.require_provenance
        else 0
    )
    structured_data_present = any(
        bool(result.structured_data) for result in relevant_results
    )
    gaps: list[str] = []
    if policy.required and len(evidence) < required_count:
        gaps.append(
            f"evidence_count_below_minimum:{len(evidence)}<{required_count}"
        )
    if missing_provenance_count:
        gaps.append(
            f"evidence_provenance_missing:{missing_provenance_count}"
        )
    if policy.require_structured_data and not structured_data_present:
        gaps.append("structured_data_missing")
    return QuestionEvidenceAssessment(
        question_id=question.question_id,
        source_task_ids=list(question.source_task_ids),
        evidence_ids=[item.evidence_id for item in evidence],
        passed=not gaps,
        evidence_count=len(evidence),
        required_count=required_count,
        missing_provenance_count=missing_provenance_count,
        structured_data_present=structured_data_present,
        gaps=gaps,
    )


async def validate_question_evidence(
    state: ResearchWorkflowState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """按研究问题执行证据数量、来源和结构化数据硬校验。"""
    del config
    plan = state.get("research_plan")
    if plan is None:
        return {"question_evidence_assessments": []}
    source_results = list(state.get("source_results") or [])
    return {
        "question_evidence_assessments": [
            _assess_question(question, source_results)
            for question in plan.questions
        ]
    }


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
    question_assessments = list(
        state.get("question_evidence_assessments") or []
    )
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
    failed_questions = [
        item for item in question_assessments if not item.passed
    ]
    for item in failed_questions:
        gaps.extend(
            f"question:{item.question_id}: {detail}"
            for detail in item.gaps
        )
    gaps = list(dict.fromkeys(gaps))

    status = deep_result.status
    error_code = deep_result.error_code
    if status == "completed" and failed_questions:
        passed_questions = [
            item for item in question_assessments if item.passed
        ]
        if passed_questions:
            status = "partial"
            error_code = "research_question_evidence_partial"
        else:
            status = "uncovered"
            error_code = "research_question_evidence_uncovered"
    elif status == "completed" and source_issues:
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
            "question_evidence_assessments": [
                item.model_dump(mode="json")
                for item in question_assessments
            ],
            "deep_agent_metadata": dict(deep_result.metadata),
        },
    )
    return {"result": result}


def build_research_workflow() -> StateGraph:
    """构建独立研究子图，DeepAgent 是其中的内部研究节点。"""
    builder = StateGraph(
        ResearchWorkflowState,
        context_schema=AgentRuntimeContext,
    )
    builder.add_node("plan_research", plan_research)
    builder.add_node("collect_sources", collect_research_sources)
    builder.add_node("deep_research", run_deep_research)
    builder.add_node(
        "validate_question_evidence",
        validate_question_evidence,
    )
    builder.add_node("finalize_research", finalize_research)
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "collect_sources")
    builder.add_edge("collect_sources", "deep_research")
    builder.add_edge("deep_research", "validate_question_evidence")
    builder.add_edge("validate_question_evidence", "finalize_research")
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
            "task_identity": dict(state.get("task_identity") or {}),
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
    "validate_question_evidence",
]
