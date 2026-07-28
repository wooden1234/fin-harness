"""Root Orchestrator V2：按任务依赖动态调度专业 Agent。"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Overwrite, Send

from agents.context_compressor import compress_context
from agents.final_answer import final_answer_node
from agents.guardrails import guardrails_edge, guardrails_node
from agents.init_turn import init_turn_node
from agents.memory_recall import memory_recall_node
from agents.orchestrator.agent_registry import get_agent_spec, invoke_agent
from agents.orchestrator.analyzer import analyze_request, heuristic_profile, latest_query
from agents.orchestrator.contracts import (
    AgentResult,
    Evidence,
    EvidencePolicy,
    QualityReport,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.evidence_quality import (
    assess_all_evidence,
    build_claim_evidence_map,
    constrained_synthesis,
    detect_claim_conflicts,
    supported_claim_ids,
)
from agents.orchestrator.error_policy import classify_error
from agents.orchestrator.task_identity import (
    ensure_task_identity,
    validate_task_plan,
)
from agents.orchestrator.planner import build_plan_from_profile
from agents.orchestrator.state import OrchestratorState
from agents.query_rewrite import query_rewrite_node
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentInput
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_v2")


def _request_budget(complexity: str) -> tuple[float, float]:
    """按请求难度返回本轮软、硬时限。"""
    prefix = {
        "simple": "SIMPLE",
        "single_capability": "SINGLE",
        "compound": "COMPOUND",
    }.get(complexity, "COMPOUND")
    return (
        float(getattr(settings, f"AGENT_V2_{prefix}_SOFT_DEADLINE_SEC")),
        float(getattr(settings, f"AGENT_V2_{prefix}_HARD_DEADLINE_SEC")),
    )


def _unit_timeouts() -> dict[str, float]:
    """返回所有执行单元的统一超时配置。"""
    return {
        "deterministic": float(settings.AGENT_V2_DETERMINISTIC_TIMEOUT_SEC),
        "agent": float(settings.AGENT_V2_AGENT_TIMEOUT_SEC),
        "workflow": float(settings.AGENT_V2_WORKFLOW_TIMEOUT_SEC),
        "tool_skill": float(settings.AGENT_V2_TOOL_SKILL_TIMEOUT_SEC),
    }


def _task_scope(runtime: Runtime[AgentRuntimeContext] | None) -> str:
    context = runtime.context if runtime is not None else None
    if context is None:
        return "default"
    return f"{context.tenant_id}:{context.conversation_id or 'run'}"


def route_after_query_rewrite(state: OrchestratorState) -> str:
    """改写成功后进入 Analyzer；无法可靠补全时直接追问。"""
    rewrite_status = str(state.get("rewrite_status") or "")
    if rewrite_status in {"success", "passthrough"}:
        return "analyze_request"
    return "clarify"


async def build_plan(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    del config
    profile = state.get("request_profile") or heuristic_profile(latest_query(state))
    if runtime is not None:
        soft_seconds, hard_seconds = _request_budget(profile.complexity)
        runtime.context.configure_budget(
            complexity=profile.complexity,
            soft_seconds=soft_seconds,
            hard_seconds=hard_seconds,
            unit_timeouts=_unit_timeouts(),
        )
    plan = validate_task_plan(
        build_plan_from_profile(profile),
        scope=_task_scope(runtime),
    )
    prior_results = (
        list(state.get("agent_results") or [])
        if profile.operation_type == "compute"
        else []
    )
    return {
        "task_plan": plan,
        "route": "plan",
        "replan_count": 0,
        "agent_results": Overwrite([]),
        # 跨轮继续过滤时，只把上一轮结果作为确定性计算输入，不拼入问题文本。
        "prior_agent_results": prior_results,
        "evidence": Overwrite([]),
        "citations": Overwrite([]),
        "claims": [],
        "claim_evidence_links": [],
        "evidence_assessments": [],
        "evidence_conflicts": [],
        "constrained_answer": None,
        "quality_report": None,
        "steps": ["orchestrator:build_plan"],
    }


def _completed_ids(state: OrchestratorState) -> set[str]:
    return {
        item.task_id
        for item in state.get("agent_results") or []
        if item.status in {"completed", "partial", "uncovered", "clarify", "failed"}
    }


def _effective_results(state: OrchestratorState) -> list[AgentResult]:
    """每个逻辑任务只保留 attempt_number 最大的结果。"""
    latest: dict[str, AgentResult] = {}
    for item in list(state.get("agent_results") or []):
        logical_id = (
            item.logical_task_id
            or str(item.metadata.get("logical_task_id") or "")
            or str(item.metadata.get("replan_of") or "")
            or item.task_id
        )
        current = latest.get(logical_id)
        if current is None or item.attempt_number >= current.attempt_number:
            latest[logical_id] = item
    return list(latest.values())


def _attach_task_metadata(result: AgentResult, task: TaskSpec) -> AgentResult:
    """把编排层任务元数据带入结果，保证 retry 关系不会在适配层丢失。"""
    logical_task_id = (
        task.logical_task_id
        or str(task.metadata.get("logical_task_id") or "")
        or str(task.metadata.get("replan_of") or "")
        or task.task_id
    )
    if task.metadata.get("replan_of") and task.logical_task_id == task.task_id:
        logical_task_id = str(task.metadata["replan_of"])
    return result.model_copy(
        update={
            "logical_task_id": logical_task_id,
            "attempt_id": task.attempt_id or None,
            "attempt_number": task.attempt_number,
            "idempotency_key": task.idempotency_key,
            "metadata": {
                **dict(result.metadata),
                **dict(task.metadata),
            }
        }
    )


def _normalize_result_error(result: AgentResult) -> AgentResult:
    """为下游统一补充 AgentResult 的错误动作。"""
    if not result.error_code or result.error_action is not None:
        return result
    decision = classify_error(result.error_code)
    return result.model_copy(
        update={
            "error_action": decision.action,
            "metadata": {
                **dict(result.metadata),
                "retryable": decision.retryable,
            },
        }
    )


def _ready_tasks(state: OrchestratorState) -> list[TaskSpec]:
    plan = state.get("task_plan")
    if plan is None:
        return []
    completed = _completed_ids(state)
    return [
        task
        for task in plan.tasks
        if task.task_id not in completed
        and all(dependency in completed for dependency in task.depends_on)
    ]


async def prepare_wave(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    del config
    context = runtime.context if runtime is not None else None
    if (
        context is not None
        and context.remaining_seconds() is not None
        and context.remaining_seconds() <= 0
    ):
        return {
            "active_task_ids": [],
            "next_action": "synthesize",
            "execution_status": "hard_timeout",
            "steps": ["orchestrator:hard_deadline_stop"],
        }
    if context is not None and context.soft_deadline_exceeded():
        return {
            "active_task_ids": [],
            "next_action": "synthesize",
            "execution_status": "soft_timeout",
            "steps": ["orchestrator:soft_deadline_degrade"],
        }
    tasks = _ready_tasks(state)
    return {
        "active_task_ids": [task.task_id for task in tasks],
        "execution_status": "running" if tasks else "evaluating",
        "steps": [f"orchestrator:prepare_wave:{len(tasks)}"],
    }


def dispatch_wave(state: OrchestratorState) -> list[Send]:
    if state.get("execution_status") in {"soft_timeout", "hard_timeout"}:
        return [Send("synthesize", {})]
    tasks = _ready_tasks(state)
    results = _effective_results(state)
    sends: list[Send] = []
    for task in tasks:
        dependencies = [
            item
            for item in results
            if item.task_id in set(task.depends_on)
        ]
        if task.agent_id == "market.compute" and not dependencies:
            dependencies = list(state.get("prior_agent_results") or [])
        sends.append(
            Send(
                "execute_task",
                {
                    "current_task": task,
                    "current_dependency_results": dependencies,
                },
            )
        )
    return sends or [Send("evaluate_results", {})]


async def execute_task(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    task = state.get("current_task")
    if task is None:
        return {"steps": ["orchestrator:execute_task:missing_task"]}
    task = ensure_task_identity(task)
    try:
        agent_kind = get_agent_spec(task.agent_id).kind
    except ValueError:
        agent_kind = "agent"
    context = runtime.context if runtime is not None else None
    configured_timeout = _unit_timeouts().get(
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
    try:
        if timeout_seconds <= 0:
            raise TimeoutError(f"run_{timeout_limit}_deadline_exceeded")
        async with asyncio.timeout(timeout_seconds):
            if context is not None:
                async with context.task_semaphore:
                    result = await invoke_agent(
                        task,
                        dependency_results=list(
                            state.get("current_dependency_results") or []
                        ),
                        config=config,
                        runtime=runtime,
                    )
            else:
                result = await invoke_agent(
                    task,
                    dependency_results=list(
                        state.get("current_dependency_results") or []
                    ),
                    config=config,
                    runtime=runtime,
                )
        result = _normalize_result_error(_attach_task_metadata(result, task))
    except TimeoutError as exc:
        error_code = {
            "hard": "run_hard_deadline_exceeded",
            "soft": "run_soft_deadline_exceeded",
        }.get(timeout_limit, "task_timeout")
        logger.warning(
            "orchestrator task timeout task_id={} timeout={} code={}",
            task.task_id,
            timeout_seconds,
            error_code,
        )
        decision = classify_error(error_code, exc)
        result = AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="failed",
            error_code=error_code,
            error_action=decision.action,
            gaps=[f"任务未在 {timeout_seconds:.1f} 秒内完成"],
            metadata={
                **dict(task.metadata),
                "retryable": decision.retryable,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("orchestrator task failed task_id={}", task.task_id)
        decision = classify_error(exc=exc)
        result = AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="failed",
            error_code=decision.error_code,
            error_action=decision.action,
            gaps=[str(exc)],
            metadata={
                **dict(task.metadata),
                "retryable": decision.retryable,
            },
        )
    return {
        "agent_results": [result],
        "evidence": list(result.evidence),
        "steps": [f"orchestrator:execute_task:{task.task_id}"],
    }


def wave_join_ready(state: OrchestratorState) -> str:
    active = set(state.get("active_task_ids") or [])
    completed = _completed_ids(state)
    return "evaluate_results" if active <= completed else END


async def evaluate_results(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    del config
    if state.get("request_profile") and state["request_profile"].missing_fields:
        return {
            "next_action": "clarify",
            "summary": "请补充需要查询的金融对象或具体目标。",
        }
    if any(
        item.error_code == "run_hard_deadline_exceeded"
        for item in _effective_results(state)
    ):
        return {
            "next_action": "synthesize",
            "execution_status": "hard_timeout",
        }
    if any(
        item.error_code == "run_soft_deadline_exceeded"
        for item in _effective_results(state)
    ):
        return {
            "next_action": "synthesize",
            "execution_status": "soft_timeout",
        }
    if any(
        item.error_action == "clarify"
        for item in _effective_results(state)
    ):
        return {
            "next_action": "clarify",
            "execution_status": "clarify_required",
        }
    if _ready_tasks(state):
        return {"next_action": "schedule", "execution_status": "running"}
    return {"next_action": "quality_gate", "execution_status": "evaluating"}


def route_after_evaluation(state: OrchestratorState) -> str:
    return str(state.get("next_action") or "synthesize")


def _evidence_policy(
    task: TaskSpec | None,
    result: AgentResult,
) -> EvidencePolicy:
    """优先使用任务覆盖配置，否则使用 Agent 注册表的默认策略。"""
    if task is not None and task.evidence_policy is not None:
        return task.evidence_policy
    try:
        return get_agent_spec(result.agent_id).evidence_policy
    except ValueError:
        return EvidencePolicy()


def _has_evidence_provenance(item: Evidence) -> bool:
    """判断证据是否具备最基本的可追溯来源信息。"""
    return bool(
        item.source_type
        and (
            item.provider
            or item.title
            or item.url
            or item.metadata
        )
    )


def _task_evidence_gaps(
    task: TaskSpec | None,
    result: AgentResult,
) -> list[str]:
    """根据任务策略检查证据数量、来源和结构化输出。"""
    policy = _evidence_policy(task, result)
    if result.status != "completed" or not policy.required:
        return []

    gaps: list[str] = []
    evidence_count = len(result.evidence)
    required_count = max(policy.min_count, 1)
    if evidence_count < required_count:
        gaps.append(
            f"{result.task_id}: evidence_count_below_minimum:"
            f"{evidence_count}<{required_count}"
        )
    if policy.require_provenance:
        missing_provenance = sum(
            1 for item in result.evidence if not _has_evidence_provenance(item)
        )
        if missing_provenance:
            gaps.append(
                f"{result.task_id}: evidence_provenance_missing:{missing_provenance}"
            )
    if policy.require_structured_data and not result.structured_data:
        gaps.append(f"{result.task_id}: structured_data_missing")
    return gaps


def detect_result_gaps(state: OrchestratorState) -> list[str]:
    """检查失败、无证据和 Agent 主动声明的结果缺口。"""
    gaps: list[str] = []
    task_by_id = {
        task.task_id: task
        for task in (state.get("task_plan").tasks if state.get("task_plan") else [])
    }
    for result in _effective_results(state):
        if result.status in {"failed", "uncovered", "clarify"}:
            gaps.append(f"{result.task_id}: {result.error_code or result.status}")
        gaps.extend(f"{result.task_id}: {item}" for item in result.gaps)
        if result.error_action in {"fallback", "fail"} and result.error_code:
            gaps.append(f"{result.task_id}: {result.error_action}:{result.error_code}")
        gaps.extend(_task_evidence_gaps(task_by_id.get(result.task_id), result))
    return list(dict.fromkeys(gaps))


def detect_evidence_conflicts(state: OrchestratorState) -> list[str]:
    """返回未解决 Claim 冲突，兼容尚未经过映射节点的直接调用。"""
    conflicts = list(state.get("evidence_conflicts") or [])
    if not conflicts:
        claims = list(state.get("claims") or [])
        links = list(state.get("claim_evidence_links") or [])
        if claims:
            conflicts = detect_claim_conflicts(claims, links)
        else:
            legacy_claims: dict[str, set[str]] = {}
            for item in state.get("evidence") or []:
                key = str(item.metadata.get("claim_key") or "")
                value = str(item.metadata.get("claim_value") or "")
                if key and value:
                    legacy_claims.setdefault(key, set()).add(value)
            return [
                f"证据冲突：{key} 出现多个值：{', '.join(sorted(values))}"
                for key, values in legacy_claims.items()
                if len(values) > 1
            ]
    return [
        f"证据冲突：{item.claim_key} 出现多个值：{', '.join(item.values)}"
        for item in conflicts
        if not item.resolved
    ]


def _effective_evidence(state: OrchestratorState) -> list[Evidence]:
    """只保留当前有效 attempt 产生的证据并按 ID 去重。"""
    deduped: dict[str, Evidence] = {}
    for result in _effective_results(state):
        for item in result.evidence:
            deduped[item.evidence_id] = item
    return list(deduped.values())


async def map_claim_evidence(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """构建 Claim—Evidence 映射并执行来源、时效和冲突评估。"""
    results = _effective_results(state)
    evidence = _effective_evidence(state)
    context = runtime.context if runtime is not None else None
    soft_remaining = (
        context.soft_remaining_seconds()
        if context is not None
        else None
    )
    allow_llm = soft_remaining is None or soft_remaining > 0
    try:
        if allow_llm and soft_remaining is not None:
            async with asyncio.timeout(soft_remaining):
                claims, links = await build_claim_evidence_map(
                    results,
                    config=config,
                    allow_llm=True,
                )
        else:
            claims, links = await build_claim_evidence_map(
                results,
                config=config,
                allow_llm=allow_llm,
            )
    except TimeoutError:
        claims, links = await build_claim_evidence_map(
            results,
            config=config,
            allow_llm=False,
        )
    assessments = assess_all_evidence(
        evidence,
        state.get("request_profile"),
    )
    conflicts = detect_claim_conflicts(claims, links)
    return {
        "claims": claims,
        "claim_evidence_links": links,
        "evidence_assessments": assessments,
        "evidence_conflicts": conflicts,
        "steps": [
            f"orchestrator:map_claim_evidence:{len(claims)}:{len(links)}"
        ],
    }


async def quality_gate(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """在综合前检查结果覆盖率和证据一致性。"""
    del config
    results = _effective_results(state)
    gaps = detect_result_gaps(state)
    conflicts = detect_evidence_conflicts(state)
    claims = list(state.get("claims") or [])
    links = list(state.get("claim_evidence_links") or [])
    assessments = list(state.get("evidence_assessments") or [])
    evidence_conflicts = list(state.get("evidence_conflicts") or [])
    supported = supported_claim_ids(
        claims,
        links,
        assessments,
        evidence_conflicts,
    )
    task_by_id = {
        task.task_id: task
        for task in (state.get("task_plan").tasks if state.get("task_plan") else [])
    }
    result_by_task = {item.task_id: item for item in results}
    required_claims = {
        claim.claim_id
        for claim in claims
        if claim.claim_type in {"fact", "calculation"}
        and _evidence_policy(
            task_by_id.get(claim.task_id),
            result_by_task.get(claim.task_id)
            or AgentResult(
                task_id=claim.task_id,
                agent_id="general_agent",
                status="completed",
            ),
        ).required
    }
    unsupported_claim_ids = sorted(required_claims - supported)
    rejected_evidence_ids = [
        item.evidence_id for item in assessments if not item.usable
    ]
    stale_evidence_ids = [
        item.evidence_id for item in assessments if item.stale
    ]
    if unsupported_claim_ids:
        gaps.extend(
            f"claim_unsupported:{claim_id}"
            for claim_id in unsupported_claim_ids
        )
    failed_ids = [
        item.task_id
        for item in results
        if item.status in {"failed", "uncovered", "clarify"}
    ]
    suggested = [
        task
        for item in results
        for task in item.suggested_tasks
    ]
    report = QualityReport(
        passed=not gaps and not conflicts,
        missing_evidence=gaps,
        conflicts=conflicts,
        unsupported_claim_ids=unsupported_claim_ids,
        stale_evidence_ids=stale_evidence_ids,
        rejected_evidence_ids=rejected_evidence_ids,
        conflict_ids=[
            item.conflict_id
            for item in evidence_conflicts
            if not item.resolved
        ],
        claim_coverage=(
            len(required_claims & supported) / len(required_claims)
            if required_claims
            else 1.0
        ),
        failed_task_ids=failed_ids,
        suggested_tasks=suggested,
        reason="质量检查通过" if not gaps and not conflicts else "存在结果缺口或证据冲突",
    )
    return {
        "quality_report": report,
        "next_action": "synthesize" if report.passed else "replan",
        "execution_status": "completed" if report.passed else "quality_failed",
        "steps": ["orchestrator:quality_gate"],
    }


def _new_replan_tasks(
    state: OrchestratorState,
    report: QualityReport,
) -> list[TaskSpec]:
    """为失败任务生成一次性补充任务，避免重复加入计划。"""
    plan = state.get("task_plan")
    if plan is None:
        return []
    existing = {task.task_id for task in plan.tasks}
    results = {item.task_id: item for item in _effective_results(state)}
    tasks = list(report.suggested_tasks)
    for failed_id in report.failed_task_ids:
        failed = results.get(failed_id)
        if failed is None:
            continue
        decision = classify_error(failed.error_code)
        if failed.error_action not in {"retry", None} and decision.action != "retry":
            continue
        if failed.error_action is None and decision.action != "retry":
            continue
        original = next((item for item in plan.tasks if item.task_id == failed_id), None)
        if original is None:
            continue
        logical_task_id = failed.logical_task_id or original.logical_task_id or failed_id
        previous_attempts = [
            item.attempt_number
            for item in [*plan.tasks, *list(state.get("agent_results") or [])]
            if (item.logical_task_id or item.task_id) == logical_task_id
        ]
        next_attempt = max(previous_attempts or [1]) + 1
        task_id = f"{logical_task_id}:retry:{next_attempt - 1}"
        if task_id in existing:
            continue
        tasks.append(
            original.model_copy(
                update={
                    "task_id": task_id,
                    "logical_task_id": logical_task_id,
                    "attempt_id": "",
                    "attempt_number": next_attempt,
                    "idempotency_key": "",
                    "depends_on": list(original.depends_on),
                    "objective": f"补充执行：{original.objective}",
                    "metadata": {**original.metadata, "replan_of": failed_id},
                }
            )
        )
    return [task for task in tasks if task.task_id not in existing]


async def replan(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    """在预算内追加补充任务；无可执行补充时转为部分回答。"""
    del config
    context = runtime.context if runtime is not None else None
    if (
        context is not None
        and context.remaining_seconds() is not None
        and context.remaining_seconds() <= 0
    ):
        return {
            "next_action": "synthesize",
            "execution_status": "hard_timeout",
            "steps": ["orchestrator:hard_deadline_stop_replan"],
        }
    if context is not None and context.soft_deadline_exceeded():
        return {
            "next_action": "synthesize",
            "execution_status": "soft_timeout",
            "steps": ["orchestrator:soft_deadline_skip_replan"],
        }
    plan = state.get("task_plan")
    report = state.get("quality_report")
    if plan is None or report is None:
        return {"next_action": "synthesize"}
    count = state.get("replan_count", 0)
    if count >= plan.max_replans:
        return {"next_action": "synthesize", "execution_status": "partial"}
    tasks = _new_replan_tasks(state, report)
    if not tasks:
        return {"next_action": "synthesize", "execution_status": "partial"}
    try:
        validated_plan = validate_task_plan(
            plan.model_copy(update={"tasks": [*plan.tasks, *tasks]}),
            scope=_task_scope(runtime),
        )
    except ValueError as exc:
        logger.warning("replan task plan validation failed: {}", str(exc))
        return {
            "next_action": "synthesize",
            "execution_status": "replan_validation_failed",
            "steps": ["orchestrator:replan_validation_failed"],
        }
    return {
        "task_plan": validated_plan,
        "replan_count": count + 1,
        "next_action": "schedule",
        "execution_status": "running",
        "steps": [f"orchestrator:replan:{count + 1}"],
    }


def route_after_quality_gate(state: OrchestratorState) -> str:
    action = str(state.get("next_action") or "synthesize")
    if action != "replan":
        return "synthesize"
    plan = state.get("task_plan")
    if plan is None or state.get("replan_count", 0) >= plan.max_replans:
        return "synthesize"
    return "replan"


def route_after_replan(state: OrchestratorState) -> str:
    return "prepare_wave" if state.get("next_action") == "schedule" else "synthesize"


async def synthesize(
    state: OrchestratorState,
    config: RunnableConfig = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
) -> dict[str, Any]:
    results = _effective_results(state)
    if not results:
        if state.get("execution_status") == "soft_timeout":
            return {
                "summary": "已达到本轮软时限，尚未获得可用结果，请缩小查询范围后重试。",
                "route": "plan",
            }
        if state.get("execution_status") == "hard_timeout":
            return {
                "summary": "已达到本轮硬时限，未完成任务已终止，请缩小查询范围后重试。",
                "route": "plan",
            }
        return {"summary": "暂时没有可用的执行结果。", "route": "plan"}
    claims = list(state.get("claims") or [])
    links = list(state.get("claim_evidence_links") or [])
    assessments = list(state.get("evidence_assessments") or [])
    evidence_conflicts = list(state.get("evidence_conflicts") or [])
    if not claims:
        claims, links = await build_claim_evidence_map(
            results,
            config=config,
            allow_llm=False,
        )
        assessments = assess_all_evidence(
            _effective_evidence(state),
            state.get("request_profile"),
        )
        evidence_conflicts = detect_claim_conflicts(claims, links)
    context = runtime.context if runtime is not None else None
    allow_llm = not (
        context is not None
        and (
            context.soft_deadline_exceeded()
            or (
                context.remaining_seconds() is not None
                and context.remaining_seconds() <= 0
            )
        )
    )
    soft_remaining = (
        context.soft_remaining_seconds()
        if context is not None
        else None
    )
    try:
        if allow_llm and soft_remaining is not None:
            async with asyncio.timeout(soft_remaining):
                constrained = await constrained_synthesis(
                    claims,
                    links,
                    assessments,
                    evidence_conflicts,
                    config=config,
                    allow_llm=True,
                )
        else:
            constrained = await constrained_synthesis(
                claims,
                links,
                assessments,
                evidence_conflicts,
                config=config,
                allow_llm=allow_llm,
            )
    except TimeoutError:
        constrained = await constrained_synthesis(
            claims,
            links,
            assessments,
            evidence_conflicts,
            config=config,
            allow_llm=False,
        )
    used_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for statement in constrained.statements
            for evidence_id in statement.evidence_ids
        )
    )
    citation_number = {
        evidence_id: index
        for index, evidence_id in enumerate(used_evidence_ids, start=1)
    }
    parts = []
    for statement in constrained.statements:
        markers = "".join(
            f"[{citation_number[evidence_id]}]"
            for evidence_id in statement.evidence_ids
            if evidence_id in citation_number
        )
        prefix = "分析判断：" if statement.statement_type == "inference" else ""
        parts.append(f"{prefix}{statement.text}{markers}")
    if constrained.caveats:
        parts.append(
            "### 限制与未解决问题\n"
            + "\n".join(f"- {item}" for item in constrained.caveats)
        )
    for result in results:
        if result.status == "completed":
            continue
        label = {
            "partial": "部分完成",
            "uncovered": "缺少证据",
            "failed": "执行失败",
            "clarify": "需要补充信息",
        }.get(result.status, result.status)
        details = "；".join(result.gaps) or result.error_code
        if details:
            parts.append(f"### {result.task_id}（{label}）\n未解决：{details}")
    report = state.get("quality_report")
    if report and not report.passed:
        unresolved = [*report.missing_evidence, *report.conflicts]
        if unresolved:
            parts.append("### 未解决问题\n" + "\n".join(f"- {item}" for item in unresolved))
    execution_status = state.get("execution_status")
    if execution_status == "soft_timeout":
        parts.append("已达到本轮软时限，系统已停止新增任务，并基于现有结果降级回答。")
    elif execution_status == "hard_timeout":
        parts.append("已达到本轮硬时限，系统已终止未完成任务；以上仅包含截止前获得的结果。")
    if not parts:
        parts.append("现有来源不足以支撑确定性结论，请补充查询范围或稍后重试。")
    evidence_by_id = {
        item.evidence_id: item
        for item in _effective_evidence(state)
    }
    return {
        "summary": "\n\n".join(parts),
        "constrained_answer": constrained,
        "citations": [
            {
                "source": item.title or item.provider,
                "snippet": item.content,
                "source_type": item.source_type,
                "sub_task_id": item.task_id,
                "evidence_id": item.evidence_id,
                **({"url": item.url} if item.url else {}),
            }
            for evidence_id in used_evidence_ids
            if (item := evidence_by_id.get(evidence_id)) is not None
        ],
        "steps": ["orchestrator:constrained_synthesize"],
    }


async def clarify(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    del state, config
    return {"summary": "请补充需要查询的金融对象或具体目标。", "route": "plan"}


def build_orchestrator_graph() -> StateGraph:
    """构建 Root Orchestrator 图。"""
    builder = StateGraph(
        OrchestratorState,
        input_schema=FinAgentInput,
        context_schema=AgentRuntimeContext,
    )
    builder.add_node("init_turn", init_turn_node)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("memory_recall", memory_recall_node)
    builder.add_node("context_compressor", compress_context)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("analyze_request", analyze_request)
    builder.add_node("build_plan", build_plan)
    builder.add_node("prepare_wave", prepare_wave)
    builder.add_node("execute_task", execute_task)
    builder.add_node("wave_join", lambda state: {"steps": ["orchestrator:wave_join"]})
    builder.add_node("evaluate_results", evaluate_results)
    builder.add_node("map_claim_evidence", map_claim_evidence)
    builder.add_node("quality_gate", quality_gate)
    builder.add_node("replan", replan)
    builder.add_node("synthesize", synthesize)
    builder.add_node("clarify", clarify)
    builder.add_node("final_answer", final_answer_node)

    builder.add_edge(START, "init_turn")
    builder.add_edge("init_turn", "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        guardrails_edge,
        {"memory_recall": "memory_recall", "final_answer": "final_answer"},
    )
    builder.add_edge("memory_recall", "context_compressor")
    builder.add_edge("context_compressor", "query_rewrite")
    builder.add_conditional_edges(
        "query_rewrite",
        route_after_query_rewrite,
        {
            "analyze_request": "analyze_request",
            "clarify": "clarify",
        },
    )
    builder.add_edge("analyze_request", "build_plan")
    builder.add_edge("build_plan", "prepare_wave")
    builder.add_conditional_edges("prepare_wave", dispatch_wave)
    builder.add_edge("execute_task", "wave_join")
    builder.add_conditional_edges(
        "wave_join",
        wave_join_ready,
        {"evaluate_results": "evaluate_results", END: END},
    )
    builder.add_conditional_edges(
        "evaluate_results",
        route_after_evaluation,
        {
            "schedule": "prepare_wave",
            "quality_gate": "map_claim_evidence",
            "clarify": "clarify",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("map_claim_evidence", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"replan": "replan", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "replan",
        route_after_replan,
        {"prepare_wave": "prepare_wave", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", "final_answer")
    builder.add_edge("clarify", "final_answer")
    builder.add_edge("final_answer", END)
    return builder


__all__ = [
    "build_orchestrator_graph",
    "dispatch_wave",
    "get_orchestrator_graph",
    "reset_orchestrator_graph_cache",
    "route_after_query_rewrite",
]


_COMPILED_GRAPHS: dict[bool, Any] = {}


def get_orchestrator_graph(*, with_checkpointer: bool = False):
    """返回 V2 编译图；默认无 Checkpoint，便于 Studio 和单测。"""
    if with_checkpointer in _COMPILED_GRAPHS:
        return _COMPILED_GRAPHS[with_checkpointer]

    builder = build_orchestrator_graph()
    if not with_checkpointer:
        compiled = builder.compile()
    else:
        from agents.checkpoint import get_checkpointer
        from app.services.memory.memory_store import get_memory_store

        compiled = builder.compile(
            checkpointer=get_checkpointer(),
            store=get_memory_store(),
        )
    _COMPILED_GRAPHS[with_checkpointer] = compiled
    return compiled


def reset_orchestrator_graph_cache() -> None:
    """清理 V2 编译图缓存，供测试和应用关闭使用。"""
    _COMPILED_GRAPHS.clear()
