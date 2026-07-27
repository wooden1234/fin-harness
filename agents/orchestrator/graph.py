"""Root Orchestrator V2：按任务依赖动态调度专业 Agent。"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Overwrite, Send

from agents.context_compressor import compress_context
from agents.final_answer import final_answer_node
from agents.guardrails import guardrails_edge, guardrails_node
from agents.memory_recall import memory_recall_node
from agents.orchestrator.agent_registry import invoke_agent
from agents.orchestrator.analyzer import analyze_request, heuristic_profile, latest_query
from agents.orchestrator.contracts import (
    AgentResult,
    QualityReport,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.planner import build_plan_from_profile
from agents.orchestrator.state import OrchestratorState
from agents.runtime_context import AgentRuntimeContext
from agents.states import FinAgentInput
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_v2")

async def build_plan(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    del config
    profile = state.get("request_profile") or heuristic_profile(latest_query(state))
    plan = build_plan_from_profile(profile)
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
    """重试任务存在时，忽略已被覆盖的原始失败结果。"""
    results = list(state.get("agent_results") or [])
    superseded = {
        str(item.metadata.get("replan_of"))
        for item in results
        if item.metadata.get("replan_of")
    }
    return [item for item in results if item.task_id not in superseded]


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
) -> dict[str, Any]:
    del config
    tasks = _ready_tasks(state)
    return {
        "active_task_ids": [task.task_id for task in tasks],
        "execution_status": "running" if tasks else "evaluating",
        "steps": [f"orchestrator:prepare_wave:{len(tasks)}"],
    }


def dispatch_wave(state: OrchestratorState) -> list[Send]:
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
    try:
        result = await invoke_agent(
            task,
            dependency_results=list(state.get("current_dependency_results") or []),
            config=config,
            runtime=runtime,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("orchestrator task failed task_id={}", task.task_id)
        result = AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            status="failed",
            error_code=type(exc).__name__,
            gaps=[str(exc)],
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
    if _ready_tasks(state):
        return {"next_action": "schedule", "execution_status": "running"}
    return {"next_action": "quality_gate", "execution_status": "evaluating"}


def route_after_evaluation(state: OrchestratorState) -> str:
    return str(state.get("next_action") or "synthesize")


def detect_result_gaps(state: OrchestratorState) -> list[str]:
    """检查失败、无证据和 Agent 主动声明的结果缺口。"""
    gaps: list[str] = []
    for result in _effective_results(state):
        if result.status in {"failed", "uncovered", "clarify"}:
            gaps.append(f"{result.task_id}: {result.error_code or result.status}")
        gaps.extend(f"{result.task_id}: {item}" for item in result.gaps)
        if result.status == "completed" and not result.evidence:
            gaps.append(f"{result.task_id}: 缺少可验证证据")
    return list(dict.fromkeys(gaps))


def detect_evidence_conflicts(state: OrchestratorState) -> list[str]:
    """根据标准化 claim 元数据检测同一事实的不同值。"""
    claims: dict[str, set[str]] = {}
    for item in state.get("evidence") or []:
        key = str(item.metadata.get("claim_key") or "")
        value = str(item.metadata.get("claim_value") or "")
        if key and value:
            claims.setdefault(key, set()).add(value)
    return [
        f"证据冲突：{key} 出现多个值：{', '.join(sorted(values))}"
        for key, values in claims.items()
        if len(values) > 1
    ]


async def quality_gate(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """在综合前检查结果覆盖率和证据一致性。"""
    del config
    results = _effective_results(state)
    gaps = detect_result_gaps(state)
    conflicts = detect_evidence_conflicts(state)
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
        task_id = f"{failed_id}:retry:{state.get('replan_count', 0) + 1}"
        if task_id in existing:
            continue
        original = next((item for item in plan.tasks if item.task_id == failed_id), None)
        if original is None:
            continue
        tasks.append(
            original.model_copy(
                update={
                    "task_id": task_id,
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
) -> dict[str, Any]:
    """在预算内追加补充任务；无可执行补充时转为部分回答。"""
    del config
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
    return {
        "task_plan": plan.model_copy(update={"tasks": [*plan.tasks, *tasks]}),
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
) -> dict[str, Any]:
    del config
    results = _effective_results(state)
    if not results:
        return {"summary": "暂时没有可用的执行结果。", "route": "plan"}
    parts = []
    for result in results:
        label = {
            "completed": "已完成",
            "partial": "部分完成",
            "uncovered": "缺少证据",
            "failed": "执行失败",
            "clarify": "需要补充信息",
        }.get(result.status, result.status)
        if result.answer or result.gaps:
            block = f"### {result.task_id}（{label}）"
            if result.answer:
                block += f"\n{result.answer}"
            if result.gaps:
                block += "\n未解决：" + "；".join(result.gaps)
            parts.append(block)
    report = state.get("quality_report")
    if report and not report.passed:
        unresolved = [*report.missing_evidence, *report.conflicts]
        if unresolved:
            parts.append("### 未解决问题\n" + "\n".join(f"- {item}" for item in unresolved))
    return {
        "summary": "\n\n".join(parts),
        "citations": [
            {
                "source": item.title or item.provider,
                "snippet": item.content,
                "source_type": item.source_type,
                "sub_task_id": item.task_id,
                **({"url": item.url} if item.url else {}),
            }
            for item in state.get("evidence") or []
        ],
        "steps": ["orchestrator:synthesize"],
    }


async def clarify(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    del state, config
    return {"summary": "请补充需要查询的金融对象或具体目标。", "route": "plan"}


def build_orchestrator_graph() -> StateGraph:
    """构建 V2 图；不修改旧的 ``agents.graph``。"""
    builder = StateGraph(
        OrchestratorState,
        input_schema=FinAgentInput,
        context_schema=AgentRuntimeContext,
    )
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("memory_recall", memory_recall_node)
    builder.add_node("context_compressor", compress_context)
    builder.add_node("analyze_request", analyze_request)
    builder.add_node("build_plan", build_plan)
    builder.add_node("prepare_wave", prepare_wave)
    builder.add_node("execute_task", execute_task)
    builder.add_node("wave_join", lambda state: {"steps": ["orchestrator:wave_join"]})
    builder.add_node("evaluate_results", evaluate_results)
    builder.add_node("quality_gate", quality_gate)
    builder.add_node("replan", replan)
    builder.add_node("synthesize", synthesize)
    builder.add_node("clarify", clarify)
    builder.add_node("final_answer", final_answer_node)

    builder.add_edge(START, "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        guardrails_edge,
        {"memory_recall": "memory_recall", "final_answer": "final_answer"},
    )
    builder.add_edge("memory_recall", "context_compressor")
    builder.add_edge("context_compressor", "analyze_request")
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
            "quality_gate": "quality_gate",
            "clarify": "clarify",
        },
    )
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
